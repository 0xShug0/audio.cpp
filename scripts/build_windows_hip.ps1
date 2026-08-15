[CmdletBinding()]
param(
    [string]$RocmPath = "",
    [string]$GpuTargets = "",
    [string]$BuildDir = "",   # default: build\hip; use a separate dir (e.g. build\hip71) for side-by-side ROCm version builds
    [string]$BuildType = "Release",
    [string]$Target = "",
    [int]$Jobs = 0,
    [switch]$ConfigureOnly,
    [switch]$Clean,
    [switch]$NoHipblasLt,    # fall back to hipBLAS (rocBLAS) GEMM
    [switch]$ForceMmq,       # route quantized matmul through GGML MMQ kernels instead of BLAS
    [switch]$NoVmm = $true,  # disable HIP virtual memory management (required on Windows iGPUs)
    [switch]$WithVmm,        # explicitly re-enable VMM
    [switch]$NoNativeCpu,    # build ggml CPU kernels without native host ISA flags (required for distribution)
    [switch]$DeploymentBuild, # compile model_specs into the binary (self-contained deployment, no model_specs/ dir needed)
    [switch]$Graphs          # enable CUDA graphs (memory-hungry on iGPUs: each cached graph reserves its own VRAM)
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Invoke-Checked {
    param(
        [Parameter(Mandatory = $true)][string]$FilePath,
        [Parameter()][string[]]$Arguments = @()
    )
    Write-Host "> $FilePath $($Arguments -join ' ')"
    & $FilePath @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed with exit code $LASTEXITCODE`: $FilePath $($Arguments -join ' ')"
    }
}

function Convert-ToCMakePath {
    param([Parameter(Mandatory = $true)][string]$Path)
    return ($Path -replace "\\", "/")
}

function Find-FirstFile {
    param([Parameter(Mandatory = $true)][string[]]$Patterns)
    foreach ($pattern in $Patterns) {
        $found = Get-ChildItem -Path $pattern -ErrorAction SilentlyContinue | Sort-Object FullName -Descending | Select-Object -First 1
        if ($null -ne $found) {
            return $found.FullName
        }
    }
    return ""
}

# --- ROCm ---
if ($RocmPath -eq "") {
    $hipPathHasClang = $env:HIP_PATH -and (
        (Test-Path (Join-Path $env:HIP_PATH "bin\clang++.exe")) -or
        (Test-Path (Join-Path $env:HIP_PATH "lib\llvm\bin\clang++.exe"))
    )
    if ($hipPathHasClang) {
        $RocmPath = (Resolve-Path $env:HIP_PATH).Path
    } else {
        $clang = Find-FirstFile @(
            "C:\Program Files\AMD\ROCm\*\bin\clang++.exe",
            "C:\TheRock\build\lib\llvm\bin\clang++.exe"
        )
        if ($clang -eq "") {
            throw "ROCm was not found. Install the AMD HIP SDK or pass -RocmPath."
        }
        $clangDir = Split-Path $clang -Parent
        if ($clangDir -like "*\lib\llvm\bin") {
            $RocmPath = (Resolve-Path (Join-Path $clangDir "..\..\..")).Path
        } else {
            $RocmPath = (Resolve-Path (Join-Path $clangDir "..")).Path
        }
    }
}
$compilerDir = @(
    (Join-Path $RocmPath "bin"),
    (Join-Path $RocmPath "lib\llvm\bin")
) | Where-Object {
    (Test-Path (Join-Path $_ "clang.exe")) -and
    (Test-Path (Join-Path $_ "clang++.exe"))
} | Select-Object -First 1
if ($null -eq $compilerDir) {
    throw "ROCm clang not found under $RocmPath\bin or $RocmPath\lib\llvm\bin"
}
$clangxx = Join-Path $compilerDir "clang++.exe"
$clangc  = Join-Path $compilerDir "clang.exe"

# --- GPU targets ---
if ($GpuTargets -eq "") {
    $amdgpuArch = Join-Path $RocmPath "bin\amdgpu-arch.exe"
    if (Test-Path $amdgpuArch) {
        $detected = & $amdgpuArch 2>$null | Where-Object { $_ -match "^gfx" } | Select-Object -Unique
        if ($detected) {
            $GpuTargets = ($detected | Sort-Object -Unique) -join ";"
        }
    }
    if ($GpuTargets -eq "") {
        throw "Could not detect GPU targets with amdgpu-arch. Pass -GpuTargets gfxXXXX."
    }
}

# --- tools ---
$cmake = Get-Command "cmake.exe" -ErrorAction SilentlyContinue
if ($null -eq $cmake) {
    throw "cmake.exe was not found on PATH"
}
$cmake = $cmake.Source

$ninjaCmd = Get-Command "ninja.exe" -ErrorAction SilentlyContinue
if ($null -ne $ninjaCmd) {
    $ninja = $ninjaCmd.Source
} else {
    $ninja = Find-FirstFile @("C:\Program Files\Microsoft Visual Studio\*\*\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja\ninja.exe")
    if ($ninja -eq "") {
        throw "ninja.exe was not found. Install ninja or Visual Studio with the CMake tools component."
    }
}

$sourceDir = Split-Path $PSScriptRoot -Parent
if ($BuildDir -eq "") {
    $BuildDir = Join-Path (Join-Path $sourceDir "build") "hip"
} elseif (-not [IO.Path]::IsPathRooted($BuildDir)) {
    $BuildDir = Join-Path $sourceDir $BuildDir
}
$buildDir = $BuildDir

$hipblasLtValue = if ($NoHipblasLt) { "OFF" } else { "ON" }
$forceMmqValue  = if ($ForceMmq) { "ON" } else { "OFF" }
$noVmmValue     = if ($WithVmm) { "OFF" } elseif ($NoVmm) { "ON" } else { "OFF" }
$nativeCpuValue = if ($NoNativeCpu) { "OFF" } else { "ON" }
$deploymentValue = if ($DeploymentBuild) { "ON" } else { "OFF" }
$graphsValue    = if ($Graphs) { "ON" } else { "OFF" }

Write-Host "ROCm:        $RocmPath"
Write-Host "GPU targets: $GpuTargets"
Write-Host "CMake:       $cmake"
Write-Host "Ninja:       $ninja"
Write-Host "Build dir:   $buildDir"
Write-Host "hipBLASLt GEMM: $hipblasLtValue, FORCE_MMQ: $forceMmqValue, NO_VMM: $noVmmValue, CUDA graphs: $graphsValue, native CPU ISA: $nativeCpuValue, deployment build: $deploymentValue"

if ($Clean) {
    Invoke-Checked $cmake @("--build", $buildDir, "--target", "clean")
    exit 0
}

$env:HIP_PATH = $RocmPath
$env:ROCM_PATH = $RocmPath
$env:PATH = @($compilerDir, (Join-Path $RocmPath "bin"), $env:PATH) -join [IO.Path]::PathSeparator

$configureArgs = @(
    "-S", $sourceDir,
    "-B", $buildDir,
    "-G", "Ninja",
    "-DCMAKE_MAKE_PROGRAM=$(Convert-ToCMakePath $ninja)",
    "-DCMAKE_BUILD_TYPE=$BuildType",
    "-DCMAKE_C_COMPILER=$(Convert-ToCMakePath $clangc)",
    "-DCMAKE_CXX_COMPILER=$(Convert-ToCMakePath $clangxx)",
    # GPU_BUILD_TARGETS/AMDGPU_TARGETS are sticky cache entries set by hip-config-amd.cmake
    # from GPU_TARGETS; without -U a previous configure's arch list wins over a new -GpuTargets.
    "-UGPU_BUILD_TARGETS",
    "-UAMDGPU_TARGETS",
    "-DENGINE_ENABLE_CUDA=OFF",
    "-DENGINE_ENABLE_HIP=ON",
    "-DENGINE_ENABLE_OPENMP=OFF",
    # ggml probes OpenMP on its own (GGML_OPENMP defaults ON) and would silently
    # pick up libomp from a vcvars environment, adding a libomp140.x86_64.dll
    # runtime dependency; force it off so the dependency list stays minimal.
    "-DGGML_OPENMP=OFF",
    "-DGPU_TARGETS=$GpuTargets",
    "-DGGML_HIP_HIPBLASLT=$hipblasLtValue",
    "-DGGML_CUDA_FORCE_MMQ=$forceMmqValue",
    "-DGGML_HIP_NO_VMM=$noVmmValue",
    "-DENGINE_ENABLE_NATIVE_CPU=$nativeCpuValue",
    "-DAUDIOCPP_DEPLOYMENT_BUILD=$deploymentValue",
    "-DENGINE_ENABLE_CUDA_GRAPHS=$graphsValue"
)

Invoke-Checked $cmake $configureArgs

if ($ConfigureOnly) {
    exit 0
}

$effectiveJobs = if ($Jobs -gt 0) { $Jobs } else { [Math]::Max(2, [Environment]::ProcessorCount) }
$buildArgs = @("--build", $buildDir, "-j", $effectiveJobs.ToString())
if ($Target -ne "") {
    $buildArgs += @("--target", $Target)
}

Write-Host "Build jobs: $effectiveJobs"
Invoke-Checked $cmake $buildArgs

Write-Host ""
Write-Host "Binaries: $buildDir\bin"
Write-Host "Note: run with '$RocmPath\bin' on PATH so the ROCm DLLs (amdhip64, hipblas, hipblaslt) are found."
