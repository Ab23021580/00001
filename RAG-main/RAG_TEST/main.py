import os
import requests
import chromadb
from pypdf import PdfReader

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PDF_FOLDER = os.path.join(BASE_DIR, "docs") if os.path.exists(os.path.join(BASE_DIR, "docs")) else r"C:\RAG_Project\docs"
DB_PATH = os.path.join(BASE_DIR, "chroma_db") if os.path.exists(os.path.join(BASE_DIR, "chroma_db")) else r"C:\RAG_Project\chroma_db"

EMBED_MODEL = "nomic-embed-text"
LLM_MODEL = "gemma3:4b"

client = chromadb.PersistentClient(path=DB_PATH)

collection = client.get_or_create_collection(
    name="rag_docs"
)


def read_pdf_docs():
    docs = []

    for filename in os.listdir(PDF_FOLDER):
        if filename.lower().endswith(".pdf"):
            filepath = os.path.join(PDF_FOLDER, filename)

            try:
                reader = PdfReader(filepath)
                text = ""

                for page in reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"

                if text.strip():
                    docs.append({
                        "source": filename,
                        "content": text
                    })
                    print(f"讀取成功：{filename}")
                else:
                    print(f"讀取不到文字：{filename}")

            except Exception as e:
                print(f"讀取失敗 {filename}：{e}")

    return docs


def chunking(text, chunk_size=900, overlap=150):
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]

        if chunk.strip():
            chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


def get_embedding(text):
    response = requests.post(
        "http://localhost:11434/api/embeddings",
        json={
            "model": EMBED_MODEL,
            "prompt": text
        }
    )

    result = response.json()
    return result["embedding"]


def build_database():
    count = collection.count()

    if count > 0:
        print(f"知識庫已存在，目前有 {count} 筆資料")
        print("如果你有更換 PDF，請刪除 chroma_db 資料夾後重建。")
        return

    docs = read_pdf_docs()

    if not docs:
        print("沒有讀到任何 PDF 內容，請確認 docs 資料夾內有 PDF。")
        return

    index = 0

    for doc in docs:
        chunks = chunking(doc["content"])

        for chunk in chunks:
            try:
                embedding = get_embedding(chunk)

                collection.add(
                    ids=[str(index)],
                    embeddings=[embedding],
                    documents=[chunk],
                    metadatas=[{
                        "source": doc["source"]
                    }]
                )

                index += 1

            except Exception as e:
                print("向量化失敗：", e)

    print(f"知識庫建立完成，共建立 {index} 個段落。")


def search_docs(question):
    embedding = get_embedding(question)

    result = collection.query(
        query_embeddings=[embedding],
        n_results=6
    )

    return result["documents"][0]


def search_docs_details(question):
    embedding = get_embedding(question)

    result = collection.query(
        query_embeddings=[embedding],
        n_results=6
    )

    documents = result["documents"][0] if result.get("documents") else []
    metadatas = result["metadatas"][0] if result.get("metadatas") else []

    details = []
    for i in range(len(documents)):
        source = metadatas[i].get("source", "未知來源") if (i < len(metadatas) and metadatas[i]) else "未知來源"
        details.append({
            "content": documents[i],
            "source": source
        })

    return details


def ask_llm(question, context):
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

    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": LLM_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.2
            }
        }
    )

    return response.json()["response"]


def chat_loop():
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
            context = "\n\n".join(docs)

            answer = ask_llm(question, context)

            print("\n回答：")
            print(answer)

        except Exception as e:
            print("發生錯誤：")
            print(e)


if __name__ == "__main__":
    build_database()
    chat_loop()