@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0"
call "%~dp0_env.bat"

REM _env.bat auto-detected BACKEND (cuda|cpu) from the NVIDIA driver + bundled exes;
REM hand it to webui.py unless the user already chose via AUDIOCPP_BACKEND.
if not defined AUDIOCPP_BACKEND (
  if /I "%BACKEND%"=="cuda" ( set "AUDIOCPP_BACKEND=gpu" ) else ( set "AUDIOCPP_BACKEND=cpu" )
)

REM Python (with gradio/requests/torch/safetensors/...) is located by _env.bat (PY).
if not exist "%PY%" (
  echo [run_webui] no Python with deps found. Looked for:
  echo   %BUNDLE%\venv\python.exe                  ^(bundle venv^)
  echo   %ROOT%\venv\python.exe                      ^(root venv^)
  echo   %ROOT%\venv\Scripts\python.exe              ^(project venv^)
  echo Install into one of them: gradio requests torch safetensors pyyaml huggingface_hub
  pause
  exit /b 1
)
echo [run_webui] python: %PY%

echo [run_webui] the WebUI starts/switches audiocpp_server on demand
echo [run_webui]   pick a model in the UI and click "load" (no need to run run_server.bat)
echo [run_webui]   backend: %AUDIOCPP_BACKEND%  (auto-detected; override with AUDIOCPP_BACKEND=gpu or cpu)
echo [run_webui] UI -^> http://127.0.0.1:7860
"%PY%" "%WEBUI_DIR%\webui.py"

endlocal
pause
