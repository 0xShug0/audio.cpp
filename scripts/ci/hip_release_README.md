# audio.cpp HIP/ROCm test prebuilt

This is an experimental test build. Requires a supported AMD GPU, a compatible
driver and the ROCm SDK @ROCM_VERSION@ runtime libraries. Linux requires Ubuntu
24.04 or newer and an AVX2/FMA/F16C-capable x64 CPU. Windows selects a CPU
implementation at runtime. `BUILD.json` lists the compiled GPU architectures
and exact source revision. Download models separately from audio-cpp on Hugging Face.

Install the matching runtime below, or use an existing matching ROCm SDK
installation. The commands select Strix Halo (`device-gfx1151`); replace that
extra with your GPU's [device extra](https://github.com/ROCm/TheRock/blob/main/RELEASES.md#supported-python-device--install-extras).
The device package supplies the required BLAS GPU kernels.

Linux, after extracting the archive (install `python3-venv` if needed):

```sh
python3 -m venv rocm-runtime
rocm-runtime/bin/python -m pip install --index-url https://stable.repo.amd.com/rocm/whl-next/ 'rocm[libraries,device-gfx1151]==@ROCM_VERSION@'
export LD_LIBRARY_PATH="$(rocm-runtime/bin/python -c 'import os,rocm_sdk; print(os.pathsep.join(dict.fromkeys(str(p.parent) for p in rocm_sdk.find_libraries("amdhip64","hipblas","hipblaslt"))))'):${LD_LIBRARY_PATH:-}"
./audiocpp_cli --list-devices
./audiocpp_server --ui --ui-management --backend hip
```

Windows PowerShell, after extracting the ZIP:

```powershell
py -3 -m venv rocm-runtime
.\rocm-runtime\Scripts\python.exe -m pip install --index-url https://stable.repo.amd.com/rocm/whl-next/ 'rocm[libraries,device-gfx1151]==@ROCM_VERSION@'
$rocm = & .\rocm-runtime\Scripts\python.exe -c "import os,rocm_sdk; print(os.pathsep.join(dict.fromkeys(str(p.parent) for p in rocm_sdk.find_libraries('amdhip64','hipblas','hipblaslt'))))"
$env:PATH = "$rocm;$env:PATH"
.\audiocpp_cli.exe --list-devices
.\audiocpp_server.exe --ui --ui-management --backend hip
```

Open the UI at http://127.0.0.1:8080. Keep the shared libraries beside the
executables. The native model manager and built-in UI are included. Models
using eSpeak additionally need eSpeak-ng and its data installed.
