@echo off
chcp 65001 >nul
setlocal
set "PYTHONIOENCODING=utf-8"
set "SCRIPT_DIR=%~dp0.."
set "PYTHONPATH=%SCRIPT_DIR%;%PYTHONPATH%"
python -m rag_qdrant.cli %*
endlocal
