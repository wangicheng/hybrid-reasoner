@echo off
setlocal

cd /d "%~dp0"

if exist ".venv\Scripts\python.exe" (
  set "PY_EXE=.venv\Scripts\python.exe"
) else if exist "venv\Scripts\python.exe" (
  set "PY_EXE=venv\Scripts\python.exe"
) else (
  set "PY_EXE=python"
)

echo Using Python: %PY_EXE%
"%PY_EXE%" scripts\run_qwen_base_tuned_test.py %*

endlocal
