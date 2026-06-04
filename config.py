"""
RAG 系統設定檔
集中管理所有配置參數
"""
import os

# 基礎路徑設定
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PDF_FOLDER = os.path.join(BASE_DIR, "docs") if os.path.exists(os.path.join(BASE_DIR, "docs")) else r"C:\RAG_Project\docs"
DB_PATH = os.path.join(BASE_DIR, "chroma_db") if os.path.exists(os.path.join(BASE_DIR, "chroma_db")) else r"C:\RAG_Project\chroma_db"

# 模型設定
EMBED_MODEL = "nomic-embed-text"
LLM_MODEL = "gemma3:4b"

# API 端點設定
OLLAMA_BASE_URL = "http://localhost:11434"
EMBEDDING_ENDPOINT = f"{OLLAMA_BASE_URL}/api/embeddings"
GENERATE_ENDPOINT = f"{OLLAMA_BASE_URL}/api/generate"

# 請求設定
REQUEST_TIMEOUT = 30  # 秒
MAX_RETRIES = 3
RETRY_DELAY = 1  # 秒

# Chunking 設定
CHUNK_SIZE = 900
CHUNK_OVERLAP = 150

# 搜尋設定
SEARCH_N_RESULTS = 6

# LLM 設定
LLM_TEMPERATURE = 0.2

# 伺服器設定
BACKEND_PORT = 8000

# CORS 設定（生產環境應改為特定網域）
ALLOWED_ORIGINS = ["*"]  # 開發環境使用，生產環境請改為 ["https://yourdomain.com"]

# Tavily 連網搜尋設定
TAVILY_API_KEY = ""  # 請填入您的 Tavily API Key（https://tavily.com）
TAVILY_ENDPOINT = "https://api.tavily.com/search"
TAVILY_MAX_RESULTS = 5
TAVILY_SEARCH_DEPTH = "basic"  # "basic" 或 "advanced"
