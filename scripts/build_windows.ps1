[CmdletBinding()]
param(
    [string]$Preset = "windows-cuda-release",
    [string]$Target = "audiocpp_cli",
    [int]$Jobs = 0,
    [switch]$ConfigureOnly,
    [switch]$Clean,
    [string]$CudaArchitectures = "auto",
    [ValidateSet("", "native", "avx2", "baseline")]
    [string]$CpuArch = "",
    [ValidateSet("ON", "OFF")]
    [string]$NativeCpu = $null,
    [ValidateSet("ON", "OFF")]
    [string]$Llamafile = $null,
    [switch]$DeploymentBuild,
    [string]$VsInstall = ""
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

function Add-PathFront {
    param([Parameter(Mandatory = $true)][string]$Path)
    if ((Test-Path $Path) -and (($env:PATH -split [IO.Path]::PathSeparator) -notcontains $Path)) {
        $env:PATH = ($Path, $env:PATH) -join [IO.Path]::PathSeparator
    }
}

function Add-EnvListFront {
    param(
        [Parameter(Mandatory = $true)][string]$Name,
        [Parameter(Mandatory = $true)][string[]]$Paths
    )

    $existing = [Environment]::GetEnvironmentVariable($Name, "Process")
    $items = @()
    foreach ($path in $Paths) {
        if ($path -and (Test-Path $path) -and ($items -notcontains $path)) {
            $items += $path
        }
    }
    if ($existing) {
        foreach ($path in ($existing -split [IO.Path]::PathSeparator)) {
            if ($path -and ($items -notcontains $path)) {
                $items += $path
            }
        }
    }
    [Environment]::SetEnvironmentVariable($Name, ($items -join [IO.Path]::PathSeparator), "Process")
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

function Find-CudaRoot {
    foreach ($root in @($env:CUDA_PATH, $env:CUDAToolkit_ROOT)) {
        if ($root -and (Test-Path (Join-Path $root "bin\nvcc.exe"))) {
            return (Resolve-Path $root).Path
        }
    }

    $nvcc = Find-FirstFile @(
        "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v*\bin\nvcc.exe",
        "C:\Program Files (x86)\NVIDIA GPU Computing Toolkit\CUDA\v*\bin\nvcc.exe"
    )
    if ($nvcc -ne "") {
        return (Resolve-Path (Join-Path (Split-Path $nvcc -Parent) "..")).Path
    }

    return ""
}

function Test-VsInstall {
    param([Parameter(Mandatory = $true)][string]$Path)
    return ($Path -ne "" -and (Test-Path (Join-Path $Path "VC\Tools\MSVC")))
}

function Find-VsInstall {
    param([string]$RequestedInstall = "")

    if ($RequestedInstall -ne "") {
        if (Test-VsInstall $RequestedInstall) {
            return (Resolve-Path $RequestedInstall).Path
        }
        throw "Requested Visual Studio Build Tools install was not usable: $RequestedInstall"
    }

    $candidates = @()
    $roots = @($env:ProgramFiles, ${env:ProgramFiles(x86)}) | Where-Object { $_ }
    foreach ($root in $roots) {
        foreach ($year in @("2026", "2022", "2019")) {
            foreach ($edition in @("BuildTools", "Community", "Professional", "Enterprise")) {
                $candidates += (Join-Path $root "Microsoft Visual Studio\$year\$edition")
            }
        }
    }

    $vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
    if (Test-Path $vswhere) {
        $found = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
        if ($LASTEXITCODE -eq 0 -and $found) {
            $candidates += $found
        }
        $found = & $vswhere -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
        if ($LASTEXITCODE -eq 0 -and $found) {
            $candidates += $found
        }
    }

    foreach ($candidate in ($candidates | Where-Object { $_ } | Select-Object -Unique)) {
        if (Test-VsInstall $candidate) {
            return (Resolve-Path $candidate).Path
        }
    }

    return ""
}

function Find-MsvcCompiler {
    param([Parameter(Mandatory = $true)][string]$VsInstall)
    return Find-FirstFile @("$VsInstall\VC\Tools\MSVC\*\bin\Hostx64\x64\cl.exe")
}

function Find-VsCMake {
    param([Parameter(Mandatory = $true)][string]$VsInstall)
    $cmake = Find-FirstFile @("$VsInstall\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe")
    if ($cmake -ne "") {
        return $cmake
    }
    $cmd = Get-Command "cmake.exe" -ErrorAction SilentlyContinue
    return if ($cmd) { $cmd.Source } else { "" }
}

function Find-VsNinja {
    param([Parameter(Mandatory = $true)][string]$VsInstall)
    $ninja = Find-FirstFile @("$VsInstall\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja\ninja.exe")
    if ($ninja -ne "") {
        return $ninja
    }
    $cmd = Get-Command "ninja.exe" -ErrorAction SilentlyContinue
    return if ($cmd) { $cmd.Source } else { "" }
}

function Find-WindowsKitTool {
    param([Parameter(Mandatory = $true)][string]$Name)
    return Find-FirstFile @(
        "C:\Program Files (x86)\Windows Kits\10\bin\*\x64\$Name",
        "C:\Program Files\Windows Kits\10\bin\*\x64\$Name"
    )
}

function Add-MsvcEnvironment {
    param(
        [Parameter(Mandatory = $true)][string]$VsInstall,
        [Parameter(Mandatory = $true)][string]$Cl,
        [Parameter(Mandatory = $true)][string]$SdkTool
    )

    $vcvars = Join-Path $VsInstall "VC\Auxiliary\Build\vcvars64.bat"
    if (Test-Path $vcvars) {
        $cmd = "`"$vcvars`" >nul && set"
        foreach ($line in (& cmd.exe /d /s /c $cmd)) {
            if ($line -match "^([^=]+)=(.*)$") {
                [Environment]::SetEnvironmentVariable($Matches[1], $Matches[2], "Process")
            }
        }
    }

    if ($Cl -notmatch "^(.*\\VC\\Tools\\MSVC\\[^\\]+)\\bin\\Hostx64\\x64\\cl\.exe$") {
        throw "Could not parse MSVC toolset root from $Cl"
    }
    $msvcRoot = $Matches[1]

    if ($SdkTool -notmatch "^(.*\\Windows Kits\\10)\\bin\\([^\\]+)\\x64\\[^\\]+\.exe$") {
        throw "Could not parse Windows SDK root from $SdkTool"
    }
    $sdkRoot = $Matches[1]
    $sdkVersion = $Matches[2]

    Add-PathFront (Join-Path $msvcRoot "bin\Hostx64\x64")
    Add-PathFront (Join-Path $sdkRoot "bin\$sdkVersion\x64")
    Add-EnvListFront "INCLUDE" @(
        (Join-Path $msvcRoot "include"),
        (Join-Path $sdkRoot "Include\$sdkVersion\ucrt"),
        (Join-Path $sdkRoot "Include\$sdkVersion\shared"),
        (Join-Path $sdkRoot "Include\$sdkVersion\um"),
        (Join-Path $sdkRoot "Include\$sdkVersion\winrt"),
        (Join-Path $sdkRoot "Include\$sdkVersion\cppwinrt")
    )
    Add-EnvListFront "LIB" @(
        (Join-Path $msvcRoot "lib\x64"),
        (Join-Path $sdkRoot "Lib\$sdkVersion\ucrt\x64"),
        (Join-Path $sdkRoot "Lib\$sdkVersion\um\x64")
    )
    Add-EnvListFront "LIBPATH" @((Join-Path $msvcRoot "lib\x64"))
}

function Resolve-CudaArchitectures {
    function Get-ReleaseCudaArchitectures {
        $nvcc = Join-Path $env:CUDA_PATH "bin\nvcc.exe"
        $supported = @()
        if (Test-Path -LiteralPath $nvcc) {
            $supported = & $nvcc --list-gpu-arch 2>$null
        }

        $wanted = @(
            @{ Compute = "compute_75"; Arch = "75-virtual" },
            @{ Compute = "compute_80"; Arch = "80-virtual" },
            @{ Compute = "compute_86"; Arch = "86-real" },
            @{ Compute = "compute_89"; Arch = "89-real" },
            @{ Compute = "compute_120"; Arch = "120a-real" },
            @{ Compute = "compute_121"; Arch = "121a-real" }
        )

        $archs = @()
        foreach ($item in $wanted) {
            if ($supported -contains $item.Compute) {
                $archs += $item.Arch
            }
        }
        if ($archs.Count -eq 0) {
            $archs = @("75-virtual", "80-virtual", "86-real")
        }
        return ($archs -join ";")
    }

    if ($CudaArchitectures -eq "default" -or $CudaArchitectures -eq "ggml-default") {
        return Get-ReleaseCudaArchitectures
    }

    if ($CudaArchitectures -ne "" -and $CudaArchitectures -ne "auto") {
        return $CudaArchitectures
    }

    $smi = Get-Command "nvidia-smi.exe" -ErrorAction SilentlyContinue
    if ($null -eq $smi) {
        return ""
    }

    $cap = (& $smi.Source --query-gpu=compute_cap --format=csv,noheader 2>$null | Select-Object -First 1).Trim()
    if ($cap -notmatch "^(\d+)\.(\d+)") {
        return ""
    }

    $major = [int]$Matches[1]
    $minor = [int]$Matches[2]
    $arch = "$major$minor"
    if ($major -ge 12) {
        return "${arch}a-real"
    }
    return "${arch}-real"
}

function Assert-OpenMpConfigured {
    param([Parameter(Mandatory = $true)][string]$BuildDir)

    $cache = Join-Path $BuildDir "CMakeCache.txt"
    if (-not (Test-Path $cache)) {
        throw "CMakeCache.txt was not created in $BuildDir"
    }
    if (-not (Select-String -Path $cache -Pattern "^GGML_OPENMP_ENABLED:INTERNAL=ON$" -Quiet)) {
        throw "OpenMP was requested, but ggml did not configure GGML_OPENMP_ENABLED=ON. Install the MSVC OpenMP component and rerun."
    }
}

function Get-CpuArchSettings {
    param([AllowEmptyString()][string]$Name)

    switch ($Name) {
        "" {
            return @{
                Label = "preset default"
                Native = $null
                CMakeArgs = @()
            }
        }
        "native" {
            return @{
                Label = "native"
                Native = "ON"
                CMakeArgs = @()
            }
        }
        "avx2" {
            return @{
                Label = "AVX2"
                Native = "OFF"
                CMakeArgs = @(
                    "-DGGML_AVX=ON",
                    "-DGGML_AVX2=ON",
                    "-DGGML_AVX512=OFF",
                    "-DGGML_AVX512_VBMI=OFF",
                    "-DGGML_AVX512_VNNI=OFF",
                    "-DGGML_AVX512_BF16=OFF",
                    "-DGGML_AVX_VNNI=OFF"
                )
            }
        }
        "baseline" {
            return @{
                Label = "baseline"
                Native = "OFF"
                CMakeArgs = @(
                    "-DGGML_AVX=OFF",
                    "-DGGML_AVX2=OFF",
                    "-DGGML_AVX512=OFF",
                    "-DGGML_AVX512_VBMI=OFF",
                    "-DGGML_AVX512_VNNI=OFF",
                    "-DGGML_AVX512_BF16=OFF",
                    "-DGGML_AVX_VNNI=OFF"
                )
            }
        }
    }
}

function Get-PresetSettings {
    param([Parameter(Mandatory = $true)][string]$Name)

    switch ($Name) {
        "windows-cpu-release" {
            return @{
                BuildType = "Release"
                BuildTests = "OFF"
                Native = "ON"
                Llamafile = "ON"
                EnableCuda = "OFF"
                EnableCudaGraphs = "OFF"
                CFlagsDebug = ""
                CxxFlagsDebug = ""
            }
        }
        "windows-cuda-debug" {
            return @{
                BuildType = "Debug"
                BuildTests = "ON"
                Native = "ON"
                Llamafile = "ON"
                EnableCuda = "ON"
                EnableCudaGraphs = "ON"
                CFlagsDebug = "/O2 /Zi"
                CxxFlagsDebug = "/O2 /Zi"
            }
        }
        "windows-cuda-release" {
            return @{
                BuildType = "Release"
                BuildTests = "OFF"
                Native = "ON"
                Llamafile = "ON"
                EnableCuda = "ON"
                EnableCudaGraphs = "ON"
                CFlagsDebug = ""
                CxxFlagsDebug = ""
            }
        }
        "windows-cuda-native-debug" {
            return @{
                BuildType = "Debug"
                BuildTests = "ON"
                Native = "ON"
                Llamafile = "ON"
                EnableCuda = "ON"
                EnableCudaGraphs = "ON"
                CFlagsDebug = "/O2 /Zi"
                CxxFlagsDebug = "/O2 /Zi"
            }
        }
        default {
            throw "Unsupported Windows preset '$Name'. Use windows-cpu-release, windows-cuda-release, windows-cuda-debug, or windows-cuda-native-debug."
        }
    }
}

$settings = Get-PresetSettings $Preset
$cpuArchSettings = Get-CpuArchSettings $CpuArch
if ($null -ne $cpuArchSettings.Native) {
    $settings.Native = $cpuArchSettings.Native
}
if (-not [string]::IsNullOrEmpty($NativeCpu)) {
    $settings.Native = $NativeCpu
}
if (-not [string]::IsNullOrEmpty($Llamafile)) {
    $settings.Llamafile = $Llamafile
}
$isCudaPreset = $settings.EnableCuda -eq "ON"

if ($isCudaPreset) {
    $cudaRoot = Find-CudaRoot
    if ($cudaRoot -eq "") {
        throw "Official CUDA Toolkit was not found. Install it so nvcc exists under C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v*\bin."
    }
    Add-PathFront (Join-Path $cudaRoot "bin")
    $env:CUDA_PATH = $cudaRoot
    $env:CUDAToolkit_ROOT = $cudaRoot
} else {
    $cudaRoot = ""
}

$vsInstall = Find-VsInstall $VsInstall
if ($vsInstall -eq "") {
    throw "Visual Studio Build Tools C++ workload was not found. Install Build Tools 2022 or newer with MSVC, Windows SDK, CMake/Ninja, and OpenMP."
}

$cl = Find-MsvcCompiler $vsInstall
$cmake = Find-VsCMake $vsInstall
$ninja = Find-VsNinja $vsInstall
$mt = Find-WindowsKitTool "mt.exe"
$rc = Find-WindowsKitTool "rc.exe"
if ($cl -eq "" -or $cmake -eq "" -or $ninja -eq "" -or $mt -eq "" -or $rc -eq "") {
    throw "Missing Build Tools component. Need cl.exe, cmake.exe, ninja.exe, mt.exe, and rc.exe."
}

Add-MsvcEnvironment $vsInstall $cl $mt
Add-PathFront (Split-Path $ninja -Parent)
$arch = if ($isCudaPreset) { Resolve-CudaArchitectures } else { "" }

if ($isCudaPreset) {
    Write-Host "CUDA: $cudaRoot"
} else {
    Write-Host "CUDA: disabled"
}
Write-Host "Visual Studio Build Tools: $vsInstall"
Write-Host "MSVC: $cl"
Write-Host "CMake: $cmake"
Write-Host "Ninja: $ninja"
Write-Host "Windows SDK: $(Split-Path $mt -Parent)"
if ($arch -ne "") {
    Write-Host "CUDA architectures: $arch"
}
Write-Host "CPU architecture profile: $($cpuArchSettings.Label)"
Write-Host "Native CPU optimization: $($settings.Native)"
Write-Host "llamafile SGEMM: $($settings.Llamafile)"
$deploymentBuildValue = if ($DeploymentBuild) { "ON" } else { "OFF" }
Write-Host "Deployment build: $deploymentBuildValue"

if ($Clean) {
    $buildDirForClean = Join-Path (Join-Path (Split-Path $PSScriptRoot -Parent) "build") $Preset
    Invoke-Checked $cmake @("--build", $buildDirForClean, "--target", "clean")
    exit 0
}

$sourceDir = Split-Path $PSScriptRoot -Parent
$buildDir = Join-Path (Join-Path $sourceDir "build") $Preset

$configureArgs = @(
    "-S", $sourceDir,
    "-B", $buildDir,
    "-G", "Ninja",
    "-DCMAKE_BUILD_TYPE=$($settings.BuildType)",
    "-DCMAKE_C_COMPILER=$(Convert-ToCMakePath $cl)",
    "-DCMAKE_CXX_COMPILER=$(Convert-ToCMakePath $cl)",
    "-DCMAKE_C_FLAGS=/utf-8",
    "-DCMAKE_CXX_FLAGS=/utf-8 /EHsc",
    "-DCMAKE_MAKE_PROGRAM=$(Convert-ToCMakePath $ninja)",
    "-DCMAKE_MT=$(Convert-ToCMakePath $mt)",
    "-DCMAKE_RC_COMPILER=$(Convert-ToCMakePath $rc)",
    "-DOpenMP_C_FLAGS=/openmp:experimental",
    "-DOpenMP_CXX_FLAGS=/openmp:experimental",
    "-DENGINE_ENABLE_CUDA=$($settings.EnableCuda)",
    "-DENGINE_ENABLE_OPENMP=ON",
    "-DENGINE_ENABLE_CUDA_GRAPHS=$($settings.EnableCudaGraphs)",
    "-DENGINE_ENABLE_VULKAN=OFF",
    "-DENGINE_ENABLE_METAL=OFF",
    "-DGGML_OPENMP=ON",
    "-DENGINE_ENABLE_NATIVE_CPU=$($settings.Native)",
    "-DENGINE_ENABLE_LLAMAFILE=$($settings.Llamafile)",
    "-DENGINE_BUILD_TESTS=$($settings.BuildTests)",
    "-DAUDIOCPP_DEPLOYMENT_BUILD=$deploymentBuildValue"
)
$configureArgs += $cpuArchSettings.CMakeArgs
if ($settings.CFlagsDebug -ne "") {
    $configureArgs += "-DCMAKE_C_FLAGS_DEBUG=$($settings.CFlagsDebug)"
}
if ($settings.CxxFlagsDebug -ne "") {
    $configureArgs += "-DCMAKE_CXX_FLAGS_DEBUG=$($settings.CxxFlagsDebug)"
}
if ($isCudaPreset) {
    $configureArgs += "-DCUDAToolkit_ROOT=$(Convert-ToCMakePath $cudaRoot)"
    $configureArgs += "-DCMAKE_CUDA_HOST_COMPILER=$(Convert-ToCMakePath $cl)"
    $configureArgs += "-DCMAKE_CUDA_FLAGS=-Xcompiler=/utf-8"
    $configureArgs += "-DOpenMP_CUDA_FLAGS=/openmp:experimental"
}
if ($isCudaPreset -and $arch -ne "") {
    $configureArgs += "-DCMAKE_CUDA_ARCHITECTURES=$arch"
} elseif ($isCudaPreset) {
    $configureArgs += @("-U", "CMAKE_CUDA_ARCHITECTURES")
}

Invoke-Checked $cmake $configureArgs
Assert-OpenMpConfigured $buildDir

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
