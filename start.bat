@echo off
setlocal

cd /d "%~dp0"

set "OLLAMA_EXE=%LocalAppData%\Programs\Ollama\ollama.exe"
set "OLLAMA_LLM_LIBRARY=cpu"
set "CUDA_VISIBLE_DEVICES=-1"
set "OLLAMA_VULKAN=false"

tasklist /FI "IMAGENAME eq ollama.exe" | find /I "ollama.exe" >nul
if errorlevel 1 (
    start "" /min "%OLLAMA_EXE%" serve
    timeout /t 5 /nobreak >nul
)

if not exist ".venv\Scripts\python.exe" (
    python -m venv .venv
    ".venv\Scripts\python.exe" -m pip install chromadb pypdf requests
)

echo Starting local RAG app at http://localhost:8000
".venv\Scripts\python.exe" app.py

pause
