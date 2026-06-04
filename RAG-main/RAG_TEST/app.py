import os
import json
import urllib.parse
from http.server import BaseHTTPRequestHandler
import socketserver
import logging

# Import RAG functions from main
import main
from config import BACKEND_PORT, ALLOWED_ORIGINS

# 設定日誌記錄（與 main 模組一致）
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

PORT = 8000

class RAGHTTPRequestHandler(BaseHTTPRequestHandler):
    def end_headers(self):
        # CORS 設定：生產環境應限制特定網域
        if ALLOWED_ORIGINS and ALLOWED_ORIGINS[0] != "*":
            origin = self.headers.get('Origin')
            if origin in ALLOWED_ORIGINS:
                self.send_header('Access-Control-Allow-Origin', origin)
        else:
            # 開發環境允許所有來源
            self.send_header('Access-Control-Allow-Origin', '*')
        
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        super().end_headers()

    def log_message(self, format, *args):
        """自訂日誌格式"""
        logger.info(f"{self.address_string()} - {format % args}")

    def do_OPTIONS(self):
        self.send_response(200)
        self.end_headers()

    def do_GET(self):
        parsed_path = urllib.parse.urlparse(self.path)
        path = parsed_path.path

        if path in ('/', '/index.html'):
            self.serve_static_file('index.html', 'text/html; charset=utf-8')
        elif path == '/api/status':
            self.handle_api_status()
        else:
            self.send_error(404, "File Not Found")

    def do_POST(self):
        parsed_path = urllib.parse.urlparse(self.path)
        path = parsed_path.path

        if path == '/api/chat':
            self.handle_api_chat()
        elif path == '/api/rebuild':
            self.handle_api_rebuild()
        else:
            self.send_error(404, "Endpoint Not Found")

    def serve_static_file(self, filename, content_type):
        try:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            filepath = os.path.join(base_dir, filename)
            
            if not os.path.exists(filepath):
                self.send_error(404, f"File {filename} not found")
                return

            with open(filepath, 'rb') as f:
                content = f.read()

            self.send_response(200)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(len(content)))
            self.end_headers()
            self.wfile.write(content)
        except Exception as e:
            self.send_error(500, f"Internal server error: {e}")

    def handle_api_status(self):
        try:
            db_count = main.collection.count()
            
            pdf_files = []
            if os.path.exists(main.PDF_FOLDER):
                pdf_files = [f for f in os.listdir(main.PDF_FOLDER) if f.lower().endswith('.pdf')]
            
            status_data = {
                "db_count": db_count,
                "pdf_count": len(pdf_files),
                "pdf_files": pdf_files,
                "llm_model": main.LLM_MODEL,
                "embed_model": main.EMBED_MODEL,
                "pdf_folder": main.PDF_FOLDER,
                "db_path": main.DB_PATH
            }
            
            response_bytes = json.dumps(status_data, ensure_ascii=False).encode('utf-8')
            self.send_response(200)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.send_header('Content-Length', str(len(response_bytes)))
            self.end_headers()
            self.wfile.write(response_bytes)
        except Exception as e:
            self.send_error(500, f"Error gathering status: {e}")

    def handle_api_chat(self):
        """處理聊天 API 請求"""
        try:
            content_length = int(self.headers.get('Content-Length', 0))
            post_data = self.rfile.read(content_length)
            req_json = json.loads(post_data.decode('utf-8'))
            
            question = req_json.get('question', '').strip()
            if not question:
                self.send_json_error(400, "問題不能為空")
                return

            logger.info(f"收到問題：{question[:50]}...")

            # Run RAG Search
            references = main.search_docs_details(question)
            
            if references is None:
                self.send_json_error(500, "搜尋失敗，請確認 Ollama 服務正常運行")
                return
            
            if not references:
                logger.warning("未找到相關參考內容")
            
            context = "\n\n".join([ref["content"] for ref in references]) if references else ""
            
            # Run LLM Ask
            answer = main.ask_llm(question, context)
            
            if answer is None:
                self.send_json_error(500, "LLM 回答失敗，請確認 Ollama 服務正常運行")
                return
            
            res_data = {
                "answer": answer,
                "references": references if references else []
            }
            
            response_bytes = json.dumps(res_data, ensure_ascii=False).encode('utf-8')
            self.send_response(200)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.send_header('Content-Length', str(len(response_bytes)))
            self.end_headers()
            self.wfile.write(response_bytes)
            
            logger.info(f"成功回應聊天請求")
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON 解析錯誤：{e}")
            self.send_json_error(400, "請求格式錯誤")
        except Exception as e:
            logger.error(f"聊天請求處理失敗：{type(e).__name__} - {e}")
            self.send_json_error(500, f"無法執行問答：{str(e)}")

    def handle_api_rebuild(self):
        """處理重建知識庫 API 請求"""
        try:
            logger.info("開始重建知識庫...")
            
            count_before = main.collection.count()
            
            if count_before > 0:
                logger.info(f"清空現有知識庫 ({count_before} 筆資料)...")
                # Delete all ids from collection to force recreation
                all_data = main.collection.get()
                if all_data and 'ids' in all_data and all_data['ids']:
                    # 分批刪除以避免記憶體問題
                    batch_size = 100
                    ids = all_data['ids']
                    for i in range(0, len(ids), batch_size):
                        batch_ids = ids[i:i + batch_size]
                        main.collection.delete(ids=batch_ids)
                    logger.info("知識庫已清空")
            
            # Run the database builder with force_rebuild flag
            result = main.build_database(force_rebuild=True)
            
            count_after = main.collection.count()
            
            if result["success"]:
                res_data = {
                    "status": "success",
                    "message": f"知識庫重建完成，共 {result['chunks_added']} 個段落。",
                    "db_count": count_after,
                    "docs_processed": result.get("docs_processed", 0),
                    "total_chunks": result.get("total_chunks", 0)
                }
                logger.info(f"知識庫重建成功：{res_data['message']}")
            else:
                res_data = {
                    "status": "partial_success" if count_after > 0 else "failed",
                    "message": result.get("message", "知識庫重建失敗"),
                    "db_count": count_after,
                    "errors": result.get("errors", [])
                }
                if result["errors"]:
                    for error in result["errors"]:
                        logger.error(error)
            
            response_bytes = json.dumps(res_data, ensure_ascii=False).encode('utf-8')
            self.send_response(200)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.send_header('Content-Length', str(len(response_bytes)))
            self.end_headers()
            self.wfile.write(response_bytes)
            
        except Exception as e:
            logger.error(f"重建知識庫失敗：{type(e).__name__} - {e}")
            self.send_json_error(500, f"重建知識庫失敗：{str(e)}")

    def send_json_error(self, status_code, message):
        res_data = {"error": message}
        response_bytes = json.dumps(res_data, ensure_ascii=False).encode('utf-8')
        self.send_response(status_code)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(response_bytes)))
        self.end_headers()
        self.wfile.write(response_bytes)

def run_server():
    """啟動 HTTP 伺服器"""
    # Set socket address reuse to avoid port collision
    socketserver.TCPServer.allow_reuse_address = True
    
    try:
        with socketserver.TCPServer(("", BACKEND_PORT), RAGHTTPRequestHandler) as httpd:
            logger.info(f"Backend Server running at http://localhost:{BACKEND_PORT}")
            print(f"\n✅ 後端伺服器已啟動：http://localhost:{BACKEND_PORT}")
            print(f"📁 PDF 資料夾：{main.PDF_FOLDER}")
            print(f"💾 資料庫路徑：{main.DB_PATH}")
            print(f"🤖 LLM 模型：{main.LLM_MODEL}")
            print(f"🔗 Embedding 模型：{main.EMBED_MODEL}")
            print("\n按 Ctrl+C 停止伺服器\n")
            
            try:
                httpd.serve_forever()
            except KeyboardInterrupt:
                logger.info("收到關閉訊號，正在停止伺服器...")
                print("\n正在關閉伺服器...")
    except OSError as e:
        if e.errno == 98 or e.errno == 10048:  # Address already in use
            logger.error(f"連接埠 {BACKEND_PORT} 已被使用")
            print(f"\n❌ 錯誤：連接埠 {BACKEND_PORT} 已被使用")
            print("請確認沒有其他程式正在使用此連接埠，或修改 config.py 中的 BACKEND_PORT 設定")
        else:
            raise


if __name__ == '__main__':
    run_server()
