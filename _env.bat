@echo off
REM _env.bat -- shared environment detection for the audio.cpp .bat launchers.
REM Called (not run) by run_webui.bat / run_server.bat / run_cli_tts.bat. It sets
REM common variables and deliberately does NOT use setlocal, so they propagate back
REM to the caller. Change detection logic here only.
REM
REM Exports: ROOT BUNDLE WEBUI_DIR PY HAS_CUDA BACKEND SERVER_EXE CLI_EXE GGUF_EXE

REM --- ROOT = this script's directory, without the trailing backslash ---
set "ROOT=%~dp0"
if "%ROOT:~-1%"=="\" set "ROOT=%ROOT:~0,-1%"

REM Dev tree: the repo root doubles as the bundle (models\ live under it; the
REM binaries live under build\). webui.py's own _find_bundle_root handles this.
set "BUNDLE=%ROOT%"
set "WEBUI_DIR=%ROOT%\webui"

REM --- Python with the deps (gradio/requests/torch/safetensors/opencc/...) ---
REM Order: explicit override, project venv (Scripts\ on Windows), then a bundle venv.
set "PY="
if defined AUDIOCPP_PYTHON if exist "%AUDIOCPP_PYTHON%" set "PY=%AUDIOCPP_PYTHON%"
if not defined PY if exist "%ROOT%\venv\Scripts\python.exe" set "PY=%ROOT%\venv\Scripts\python.exe"
if not defined PY if exist "%ROOT%\venv\python.exe"         set "PY=%ROOT%\venv\python.exe"
if not defined PY if exist "%BUNDLE%\venv\Scripts\python.exe" set "PY=%BUNDLE%\venv\Scripts\python.exe"
if not defined PY if exist "%BUNDLE%\venv\python.exe"         set "PY=%BUNDLE%\venv\python.exe"

REM --- CUDA present? (NVIDIA driver installs nvcuda.dll in System32) ---
set "HAS_CUDA="
if exist "%SystemRoot%\System32\nvcuda.dll" set "HAS_CUDA=1"

REM --- Locate the from-source binaries. The default Visual Studio generator nests
REM them in build\bin\Release (multi-config); Ninja/Makefiles use build\bin. ---
set "BIN="
if exist "%ROOT%\build\bin\Release\audiocpp_server.exe" set "BIN=%ROOT%\build\bin\Release"
if not defined BIN if exist "%ROOT%\build\bin\audiocpp_server.exe" set "BIN=%ROOT%\build\bin"
if defined BIN set "SERVER_EXE=%BIN%\audiocpp_server.exe"
if defined BIN set "CLI_EXE=%BIN%\audiocpp_cli.exe"
if defined BIN set "GGUF_EXE=%BIN%\audiocpp_gguf.exe"

REM --- BACKEND: read the actual build's GGML_CUDA flag from its CMakeCache, so we
REM never advertise a GPU backend a CPU-only build can't serve. cuda when ON, else cpu. ---
set "BACKEND=cpu"
if exist "%ROOT%\build\CMakeCache.txt" (
    for /f "tokens=2 delims==" %%A in ('findstr /b /c:"GGML_CUDA:BOOL" "%ROOT%\build\CMakeCache.txt" 2^>nul') do (
        if /I "%%A"=="ON" set "BACKEND=cuda"
    )
)
