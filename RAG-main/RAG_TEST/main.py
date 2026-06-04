"""
RAG 知識庫核心模組
提供 PDF 讀取、向量化、知識庫建構與查詢功能
"""
import os
import time
import logging
import requests
from requests.exceptions import RequestException, Timeout, ConnectionError
from typing import List, Dict, Optional, Any
import chromadb
from chromadb.config import Settings
from pypdf import PdfReader
from pypdf.errors import PdfReadError as PyPdfError

# 導入設定
from config import (
    PDF_FOLDER, DB_PATH, EMBED_MODEL, LLM_MODEL,
    EMBEDDING_ENDPOINT, GENERATE_ENDPOINT,
    REQUEST_TIMEOUT, MAX_RETRIES, RETRY_DELAY,
    CHUNK_SIZE, CHUNK_OVERLAP, SEARCH_N_RESULTS, LLM_TEMPERATURE
)

# 設定日誌記錄
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# 初始化 ChromaDB 客戶端
try:
    client = chromadb.PersistentClient(path=DB_PATH)
    collection = client.get_or_create_collection(name="rag_docs")
    logger.info(f"ChromaDB 初始化成功，路徑：{DB_PATH}")
except Exception as e:
    logger.error(f"ChromaDB 初始化失敗：{e}")
    raise


def read_pdf_docs() -> List[Dict[str, str]]:
    """
    讀取 PDF 資料夾中的所有 PDF 檔案
    
    Returns:
        List[Dict]: 包含 source 和 content 的文件列表
    """
    docs = []
    
    if not os.path.exists(PDF_FOLDER):
        logger.error(f"PDF 資料夾不存在：{PDF_FOLDER}")
        return docs

    pdf_files = [f for f in os.listdir(PDF_FOLDER) if f.lower().endswith(".pdf")]
    
    if not pdf_files:
        logger.warning(f"PDF 資料夾中沒有找到 PDF 檔案：{PDF_FOLDER}")
        return docs
    
    logger.info(f"開始讀取 {len(pdf_files)} 個 PDF 檔案")

    for filename in pdf_files:
        filepath = os.path.join(PDF_FOLDER, filename)

        try:
            reader = PdfReader(filepath)
            text_parts = []

            for i, page in enumerate(reader.pages):
                try:
                    page_text = page.extract_text()
                    if page_text:
                        text_parts.append(page_text)
                except Exception as page_err:
                    logger.warning(f"PDF {filename} 第 {i+1} 頁提取失敗：{page_err}")

            text = "\n".join(text_parts)

            if text.strip():
                docs.append({
                    "source": filename,
                    "content": text
                })
                logger.info(f"讀取成功：{filename} ({len(text)} 字元)")
            else:
                logger.warning(f"讀取不到文字：{filename}")

        except PyPdfError as e:
            logger.error(f"PDF 格式錯誤 {filename}：{e}")
        except PermissionError as e:
            logger.error(f"無權限讀取 {filename}：{e}")
        except Exception as e:
            logger.error(f"讀取失敗 {filename}：{type(e).__name__} - {e}")

    logger.info(f"成功讀取 {len(docs)} 個 PDF 檔案")
    return docs


def chunking(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """
    將文字分割成 chunks
    
    Args:
        text: 要分割的文字
        chunk_size: 每個 chunk 的大小（字元數）
        overlap: chunk 之間的重疊字元數
    
    Returns:
        List[str]: chunk 列表
    """
    if not text or not text.strip():
        return []
    
    chunks = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = start + chunk_size
        
        # 嘗試在句子邊界處切割（尋找句號、問號、感嘆號或換行）
        if end < text_len:
            # 優先尋找段落結束
            for sep in ['\n\n', '\n', '。', '！', '？', '.', '!', '?']:
                last_sep = text.rfind(sep, start, end)
                if last_sep > start + chunk_size // 2:  # 至少在 chunk 的中間之後
                    end = last_sep + len(sep)
                    break
        
        chunk = text[start:end]

        if chunk.strip():
            chunks.append(chunk.strip())

        start += chunk_size - overlap

    return chunks


def get_embedding(text: str) -> Optional[List[float]]:
    """
    取得文字的嵌入向量，包含錯誤處理和重試機制
    
    Args:
        text: 要嵌入的文字
    
    Returns:
        List[float]: 嵌入向量，失敗時返回 None
    """
    if not text or not text.strip():
        logger.warning("嘗試嵌入空文字")
        return None
    
    payload = {
        "model": EMBED_MODEL,
        "prompt": text
    }
    
    for attempt in range(MAX_RETRIES):
        try:
            response = requests.post(
                EMBEDDING_ENDPOINT,
                json=payload,
                timeout=REQUEST_TIMEOUT
            )
            
            # 檢查 HTTP 狀態碼
            if response.status_code != 200:
                logger.error(f"Embedding API 返回錯誤狀態碼：{response.status_code}")
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAY * (attempt + 1))
                    continue
                return None
            
            result = response.json()
            
            if "embedding" not in result:
                logger.error(f"Embedding 回應格式錯誤：{result}")
                return None
                
            return result["embedding"]
            
        except Timeout:
            logger.warning(f"Embedding 請求超時（第 {attempt + 1}/{MAX_RETRIES} 次）")
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY * (attempt + 1))
                
        except ConnectionError as e:
            logger.error(f"連接錯誤：{e}")
            return None
            
        except RequestException as e:
            logger.error(f"請求失敗（第 {attempt + 1}/{MAX_RETRIES} 次）：{type(e).__name__} - {e}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY * (attempt + 1))
                
        except Exception as e:
            logger.error(f"未預期的錯誤：{type(e).__name__} - {e}")
            return None
    
    logger.error(f"Embedding 請求在 {MAX_RETRIES} 次重試後仍失敗")
    return None


def build_database(force_rebuild: bool = False) -> Dict[str, Any]:
    """
    建立或重建知識庫
    
    Args:
        force_rebuild: 是否強制重建（即使知識庫已存在）
    
    Returns:
        Dict: 包含建構結果的資訊
    """
    result = {
        "success": False,
        "chunks_added": 0,
        "docs_processed": 0,
        "errors": []
    }
    
    count = collection.count()

    if count > 0 and not force_rebuild:
        msg = f"知識庫已存在，目前有 {count} 筆資料。如需重建請使用 force_rebuild=True"
        logger.info(msg)
        result["message"] = msg
        return result

    # 如果需要重建，先清空現有資料
    if count > 0 and force_rebuild:
        logger.info(f"清空現有知識庫 ({count} 筆資料)...")
        try:
            all_data = collection.get()
            if all_data and 'ids' in all_data and all_data['ids']:
                # 分批刪除以避免記憶體問題
                batch_size = 100
                ids = all_data['ids']
                for i in range(0, len(ids), batch_size):
                    batch_ids = ids[i:i + batch_size]
                    collection.delete(ids=batch_ids)
                logger.info("知識庫已清空")
        except Exception as e:
            error_msg = f"清空知識庫失敗：{e}"
            logger.error(error_msg)
            result["errors"].append(error_msg)
            return result

    docs = read_pdf_docs()

    if not docs:
        msg = "沒有讀到任何 PDF 內容，請確認 docs 資料夾內有 PDF。"
        logger.warning(msg)
        result["message"] = msg
        return result
    
    result["docs_processed"] = len(docs)
    total_chunks = 0
    success_chunks = 0

    logger.info(f"開始處理 {len(docs)} 個文件...")
    
    for doc_idx, doc in enumerate(docs, 1):
        logger.info(f"處理文件 {doc_idx}/{len(docs)}: {doc['source']}")
        chunks = chunking(doc["content"])
        total_chunks += len(chunks)
        
        if not chunks:
            logger.warning(f"文件 {doc['source']} 未產生任何 chunks")
            continue

        # 批次處理嵌入（每 10 個顯示一次進度）
        for chunk_idx, chunk in enumerate(chunks):
            embedding = get_embedding(chunk)
            
            if embedding is None:
                error_msg = f"向量化失敗：{doc['source']} - chunk {chunk_idx}"
                logger.error(error_msg)
                result["errors"].append(error_msg)
                continue

            try:
                global_index = success_chunks
                
                collection.add(
                    ids=[f"{doc['source']}_{global_index}"],
                    embeddings=[embedding],
                    documents=[chunk],
                    metadatas=[{
                        "source": doc["source"]
                    }]
                )
                
                success_chunks += 1
                
                # 每 10 個 chunk 顯示進度
                if (success_chunks % 10 == 0) or (success_chunks == len(chunks)):
                    logger.info(f"進度：{success_chunks}/{total_chunks} chunks")

            except Exception as e:
                error_msg = f"加入知識庫失敗：{type(e).__name__} - {e}"
                logger.error(error_msg)
                result["errors"].append(error_msg)

    result["success"] = success_chunks > 0
    result["chunks_added"] = success_chunks
    result["total_chunks"] = total_chunks
    
    msg = f"知識庫建立完成，共建立 {success_chunks}/{total_chunks} 個段落"
    logger.info(msg)
    result["message"] = msg
    
    return result


def search_docs(question: str) -> Optional[List[str]]:
    """
    搜尋知識庫文件
    
    Args:
        question: 使用者問題
    
    Returns:
        List[str]: 相關文件內容列表，失敗時返回 None
    """
    if not question or not question.strip():
        logger.warning("搜尋問題為空")
        return None
    
    embedding = get_embedding(question)
    
    if embedding is None:
        logger.error("無法取得問題的嵌入向量")
        return None

    try:
        result = collection.query(
            query_embeddings=[embedding],
            n_results=SEARCH_N_RESULTS
        )

        if not result.get("documents") or not result["documents"][0]:
            logger.warning("未找到相關文件")
            return []

        return result["documents"][0]
        
    except Exception as e:
        logger.error(f"搜尋失敗：{type(e).__name__} - {e}")
        return None


def search_docs_details(question: str) -> Optional[List[Dict[str, str]]]:
    """
    搜尋知識庫文件（含詳細資訊）
    
    Args:
        question: 使用者問題
    
    Returns:
        List[Dict]: 包含 content 和 source 的文件列表，失敗時返回 None
    """
    if not question or not question.strip():
        logger.warning("搜尋問題為空")
        return None
    
    embedding = get_embedding(question)
    
    if embedding is None:
        logger.error("無法取得問題的嵌入向量")
        return None

    try:
        result = collection.query(
            query_embeddings=[embedding],
            n_results=SEARCH_N_RESULTS
        )

        documents = result.get("documents", [[]])[0] if result.get("documents") else []
        metadatas = result.get("metadatas", [[]])[0] if result.get("metadatas") else []

        details = []
        for i in range(len(documents)):
            if not documents[i]:
                continue
                
            source = "未知來源"
            if i < len(metadatas) and metadatas[i]:
                source = metadatas[i].get("source", "未知來源")
            
            details.append({
                "content": documents[i],
                "source": source
            })

        return details
        
    except Exception as e:
        logger.error(f"搜尋失敗：{type(e).__name__} - {e}")
        return None


def ask_llm(question: str, context: str) -> Optional[str]:
    """
    使用 LLM 回答問題
    
    Args:
        question: 使用者問題
        context: 搜尋到的參考內容
    
    Returns:
        str: LLM 回答，失敗時返回 None
    """
    if not context or not context.strip():
        logger.warning("參考內容為空")
    
    prompt = f"""
你是一位專業的本機 RAG 知識庫助理。

請根據下方「參考內容」回答使用者問題。

回答要求：
1. 使用繁體中文。
2. 先用一段話直接回答問題。
3. 再用條列式整理重點。
4. 不要顯示檔案路徑。
5. 不要提到 ChromaDB、Embedding、向量資料庫。
6. 不要亂編參考內容沒有的資訊。
7. 如果參考內容只有部分相關，請說「目前知識庫只找到部分相關內容」，並整理可用資訊。
8. 只有在完全沒有相關內容時，才回答「文件中沒有相關資訊」。

參考內容：
{context}

使用者問題：
{question}

請開始回答：
"""

    payload = {
        "model": LLM_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": LLM_TEMPERATURE
        }
    }
    
    for attempt in range(MAX_RETRIES):
        try:
            response = requests.post(
                GENERATE_ENDPOINT,
                json=payload,
                timeout=REQUEST_TIMEOUT * 2  # LLM 生成需要更長時間
            )
            
            if response.status_code != 200:
                logger.error(f"LLM API 返回錯誤狀態碼：{response.status_code}")
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAY * (attempt + 1))
                    continue
                return None
            
            result = response.json()
            
            if "response" not in result:
                logger.error(f"LLM 回應格式錯誤：{result}")
                return None
                
            return result["response"]
            
        except Timeout:
            logger.warning(f"LLM 請求超時（第 {attempt + 1}/{MAX_RETRIES} 次）")
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY * (attempt + 1))
                
        except ConnectionError as e:
            logger.error(f"連接錯誤：{e}")
            return None
            
        except RequestException as e:
            logger.error(f"請求失敗（第 {attempt + 1}/{MAX_RETRIES} 次）：{type(e).__name__} - {e}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY * (attempt + 1))
                
        except Exception as e:
            logger.error(f"未預期的錯誤：{type(e).__name__} - {e}")
            return None
    
    logger.error(f"LLM 請求在 {MAX_RETRIES} 次重試後仍失敗")
    return None


def chat_loop():
    """互動式對話迴圈（命令列版本）"""
    print("\n歡迎使用本機 RAG 知識庫")
    print("輸入 q 可以離開")

    while True:
        question = input("\n請輸入問題：").strip()

        if question.lower() == "q":
            break

        if not question:
            print("請輸入問題")
            continue

        try:
            docs = search_docs(question)
            
            if docs is None:
                print("發生錯誤：無法執行搜尋")
                continue
                
            if not docs:
                print("未找到相關內容")
                continue
            
            context = "\n\n".join(docs)

            answer = ask_llm(question, context)
            
            if answer:
                print("\n回答：")
                print(answer)
            else:
                print("無法取得回答")

        except Exception as e:
            logger.error(f"對話發生錯誤：{type(e).__name__} - {e}")
            print("發生錯誤，請查看日誌以獲取詳細資訊")


if __name__ == "__main__":
    # 檢查是否需要重建資料庫
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--rebuild":
        logger.info("開始重建知識庫...")
        result = build_database(force_rebuild=True)
        if result["success"]:
            logger.info(result["message"])
        else:
            logger.error(result.get("message", "知識庫建立失敗"))
            if result["errors"]:
                for error in result["errors"]:
                    logger.error(error)
    else:
        # 如果知識庫為空，自動建立
        count = collection.count()
        if count == 0:
            logger.info("知識庫為空，開始建立...")
            build_database()
    
    # 啟動對話迴圈
    chat_loop()