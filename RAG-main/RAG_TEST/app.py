import os
import json
import urllib.parse
from http.server import BaseHTTPRequestHandler
import socketserver

# Import RAG functions from main
import main

PORT = 8000

class RAGHTTPRequestHandler(BaseHTTPRequestHandler):
    def end_headers(self):
        # Allow CORS for development/flexibility
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        super().end_headers()

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
        try:
            content_length = int(self.headers.get('Content-Length', 0))
            post_data = self.rfile.read(content_length)
            req_json = json.loads(post_data.decode('utf-8'))
            
            question = req_json.get('question', '').strip()
            if not question:
                self.send_json_error(400, "問題不能為空")
                return

            # Run RAG Search
            references = main.search_docs_details(question)
            context = "\n\n".join([ref["content"] for ref in references])
            
            # Run LLM Ask
            answer = main.ask_llm(question, context)
            
            res_data = {
                "answer": answer,
                "references": references
            }
            
            response_bytes = json.dumps(res_data, ensure_ascii=False).encode('utf-8')
            self.send_response(200)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.send_header('Content-Length', str(len(response_bytes)))
            self.end_headers()
            self.wfile.write(response_bytes)
        except Exception as e:
            self.send_json_error(500, f"無法執行問答：{str(e)}")

    def handle_api_rebuild(self):
        try:
            count_before = main.collection.count()
            if count_before > 0:
                # Delete all ids from collection to force recreation
                all_data = main.collection.get()
                if all_data and 'ids' in all_data and all_data['ids']:
                    main.collection.delete(ids=all_data['ids'])
            
            # Run the database builder
            main.build_database()
            
            count_after = main.collection.count()
            
            res_data = {
                "status": "success",
                "message": f"知識庫重建完成，共 {count_after} 個段落。",
                "db_count": count_after
            }
            
            response_bytes = json.dumps(res_data, ensure_ascii=False).encode('utf-8')
            self.send_response(200)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.send_header('Content-Length', str(len(response_bytes)))
            self.end_headers()
            self.wfile.write(response_bytes)
        except Exception as e:
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
    # Set socket address reuse to avoid port collision
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("", PORT), RAGHTTPRequestHandler) as httpd:
        print(f"Backend Server running at http://localhost:{PORT}")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down server...")

if __name__ == '__main__':
    run_server()
