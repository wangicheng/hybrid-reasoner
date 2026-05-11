@echo off
REM Hybrid Reasoner Web Server Startup Script

echo ==========================================
echo  Hybrid Reasoner - Web Server
echo ==========================================

REM Activate virtual environment
call .venv\Scripts\activate

REM Install dependencies if needed
echo Checking dependencies...
pip install -r requirements.txt

REM Run the web API server
echo Starting server at http://localhost:8000
python -m uvicorn src.web_api:app --host 0.0.0.0 --port 8000 --reload
pause
