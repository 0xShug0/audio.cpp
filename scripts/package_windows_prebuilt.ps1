[CmdletBinding()]
param(
    [ValidateSet("all", "cpu", "cuda")]
    [string]$Package = "all",
    [ValidateSet("fast", "balance", "portable")]
    [string]$Profile = "balance",
    [int]$Jobs = 0,
    [string]$OutputDir = "",
    [string]$CudaArchitectures = "default",
    [string]$VsInstall = "",
    [switch]$SkipCudaRuntimeDlls
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

function Find-CudaRoot {
    foreach ($root in @($env:CUDA_PATH, $env:CUDAToolkit_ROOT)) {
        if ($root -and (Test-Path (Join-Path $root "bin\nvcc.exe"))) {
            return (Resolve-Path $root).Path
        }
    }
    $roots = Get-ChildItem "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA" -Directory -ErrorAction SilentlyContinue |
        Sort-Object Name -Descending
    foreach ($root in $roots) {
        if (Test-Path (Join-Path $root.FullName "bin\nvcc.exe")) {
            return $root.FullName
        }
    }
    return ""
}

function Find-VcRedistDir {
    param([Parameter(Mandatory = $true)][string]$Name)

    $root = "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Redist\MSVC"
    $found = Get-ChildItem $root -Directory -ErrorAction SilentlyContinue |
        Sort-Object Name -Descending |
        ForEach-Object { Join-Path $_.FullName "x64\$Name" } |
        Where-Object { Test-Path $_ } |
        Select-Object -First 1
    if (-not $found) {
        throw "Could not find MSVC redist directory '$Name'. Install Visual Studio Build Tools C++ redistributables."
    }
    return $found
}

function Copy-RequiredDll {
    param(
        [Parameter(Mandatory = $true)][string]$Name,
        [Parameter(Mandatory = $true)][string[]]$SearchDirs,
        [Parameter(Mandatory = $true)][string]$Destination
    )

    foreach ($dir in $SearchDirs) {
        if (-not $dir) {
            continue
        }
        $path = Join-Path $dir $Name
        if (Test-Path -LiteralPath $path) {
            Copy-Item -LiteralPath $path -Destination $Destination -Force
            return (Join-Path $Destination $Name)
        }
    }
    throw "Could not find required DLL '$Name'. Searched: $($SearchDirs -join '; ')"
}

function Copy-TreeContents {
    param(
        [Parameter(Mandatory = $true)][string]$Source,
        [Parameter(Mandatory = $true)][string]$Destination
    )

    if (Test-Path -LiteralPath $Destination) {
        Remove-Item -LiteralPath $Destination -Recurse -Force
    }
    New-Item -ItemType Directory -Force -Path $Destination | Out-Null
    Get-ChildItem -LiteralPath $Source -Force | Copy-Item -Destination $Destination -Recurse -Force
}

function Write-PackageReadme {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][string]$Kind,
        [Parameter(Mandatory = $true)][hashtable]$ProfileSettings,
        [Parameter(Mandatory = $true)][bool]$IncludesCudaRuntimeDlls
    )

    $profileLabel = [string]$ProfileSettings["Label"]
    $profileDescription = [string]$ProfileSettings["Description"]
    $profileText = @(
        "",
        "## Performance Profile",
        "",
        "This package uses the $profileLabel profile.",
        "",
        $profileDescription,
        ""
    ) -join "`r`n"

    if ($Kind -eq "cuda") {
        $cudaRuntimeBullet = if ($IncludesCudaRuntimeDlls) { "- CUDA runtime DLLs required by this build" } else { "" }
        $cudaRuntimeNote = if ($IncludesCudaRuntimeDlls) {
            "The CUDA Toolkit and Visual Studio Build Tools are not required to run this package."
        } else {
            "This package does not include CUDA runtime DLLs. Download the matching `audiocpp-windows-cuda-runtime` package and place its DLLs next to these executables, or provide the CUDA runtime DLLs through `PATH`."
        }
        $cudaBodyLines = @(
            "# audio.cpp Windows CUDA Prebuilt",
            "",
            "This package contains:",
            "",
            "- `audiocpp_cli.exe`",
            "- `audiocpp_server.exe`",
            $cudaRuntimeBullet,
            "- MSVC and OpenMP runtime DLLs required by this build",
            "",
            "## Requirements",
            "",
            "- 64-bit Windows",
            "- NVIDIA GPU with a compatible NVIDIA driver",
            "- Model files downloaded separately",
            "",
            $cudaRuntimeNote,
            "This CUDA build also includes the CPU backend, so it can run with `--backend cpu` or `--backend cuda`.",
            $profileText
        ) | Where-Object { $_ -ne "" }
        $body = $cudaBodyLines -join "`r`n"
        $body += @'

## Quick Use

```powershell
.\audiocpp_cli.exe --help
.\audiocpp_cli.exe --backend cuda --task tts --family <family> --model C:\path\to\model [options]
.\audiocpp_cli.exe --backend cpu --task tts --family <family> --model C:\path\to\model [options]
```

Server:

```powershell
.\audiocpp_server.exe --config C:\path\to\server.json
```

## Notes

- Models are not bundled.
- Keep the DLL files next to the `.exe` files.
- If CUDA startup fails, update the NVIDIA driver first.
'@
    } else {
        $body = @'
# audio.cpp Windows CPU Prebuilt

This package contains:

- `audiocpp_cli.exe`
- `audiocpp_server.exe`
- MSVC and OpenMP runtime DLLs required by this build

## Requirements

- 64-bit Windows
- Model files downloaded separately

The CUDA Toolkit, NVIDIA GPU, and Visual Studio Build Tools are not required to run this package.

'@ + $profileText + @'

## Quick Use

```powershell
.\audiocpp_cli.exe --help
.\audiocpp_cli.exe --backend cpu --task tts --family <family> --model C:\path\to\model [options]
```

Server:

```powershell
.\audiocpp_server.exe --config C:\path\to\server.json
```

## Notes

- Models are not bundled.
- Keep the DLL files next to the `.exe` files.
'@
    }
    Set-Content -LiteralPath $Path -Value $body -Encoding UTF8
}

function Get-ProfileSettings {
    param([Parameter(Mandatory = $true)][string]$Name)

    switch ($Name) {
        "fast" {
            return @{
                Label = "fast"
                CpuArch = "native"
                Llamafile = "ON"
                Description = "Fastest on machines similar to the build machine. This can use native CPU instructions such as AVX512 when available, so it is not the safest choice for broad public distribution."
            }
        }
        "balance" {
            return @{
                Label = "balance"
                CpuArch = "avx2"
                Llamafile = "ON"
                Description = "Recommended default for most modern Windows PCs. It avoids native/AVX512 CPU selection while still using AVX2-class CPU kernels. It is usually slower than fast on high-end CPUs, but much more compatible."
            }
        }
        "portable" {
            return @{
                Label = "portable"
                CpuArch = "baseline"
                Llamafile = "OFF"
                Description = "Best compatibility. It avoids native CPU tuning, AVX/AVX2/AVX512 selection, and llamafile SGEMM. It is the slowest profile, but the least likely to fail on older or unusual Windows machines."
            }
        }
    }
}

function New-PrebuiltPackage {
    param(
        [Parameter(Mandatory = $true)][string]$Preset,
        [Parameter(Mandatory = $true)][ValidateSet("cpu", "cuda")][string]$Kind
    )

    $buildArgs = @(
        "-Preset", $Preset,
        "-Target", "audiocpp_cli",
        "-DeploymentBuild",
        "-Jobs", $Jobs.ToString(),
        "-CpuArch", $profileSettings.CpuArch,
        "-Llamafile", $profileSettings.Llamafile
    )
    if ($Kind -eq "cuda") {
        $buildArgs += @("-CudaArchitectures", $CudaArchitectures)
    }
    if ($VsInstall -ne "") {
        $buildArgs += @("-VsInstall", $VsInstall)
    }
    Invoke-Checked "powershell.exe" (@("-NoProfile", "-ExecutionPolicy", "Bypass", "-File", $buildScript) + $buildArgs)

    $buildArgs = @(
        "-Preset", $Preset,
        "-Target", "audiocpp_server",
        "-DeploymentBuild",
        "-Jobs", $Jobs.ToString(),
        "-CpuArch", $profileSettings.CpuArch,
        "-Llamafile", $profileSettings.Llamafile
    )
    if ($Kind -eq "cuda") {
        $buildArgs += @("-CudaArchitectures", $CudaArchitectures)
    }
    if ($VsInstall -ne "") {
        $buildArgs += @("-VsInstall", $VsInstall)
    }
    Invoke-Checked "powershell.exe" (@("-NoProfile", "-ExecutionPolicy", "Bypass", "-File", $buildScript) + $buildArgs)

    $sourceBin = Join-Path (Join-Path $repoRoot "build") "$Preset\bin"
    if (-not (Test-Path (Join-Path $sourceBin "audiocpp_cli.exe")) -or -not (Test-Path (Join-Path $sourceBin "audiocpp_server.exe"))) {
        throw "Expected binaries were not found in $sourceBin"
    }

    $packageName = "audiocpp-windows-$Kind-$Profile"
    $stageDir = Join-Path $OutputDir $packageName
    Copy-TreeContents $sourceBin $stageDir

    $crtDir = Find-VcRedistDir "Microsoft.VC143.CRT"
    $ompDir = Find-VcRedistDir "Microsoft.VC143.OpenMP"
    Copy-RequiredDll "MSVCP140.dll" @($crtDir) $stageDir | Out-Null
    Copy-RequiredDll "VCRUNTIME140.dll" @($crtDir) $stageDir | Out-Null
    Copy-RequiredDll "VCRUNTIME140_1.dll" @($crtDir) $stageDir | Out-Null
    Copy-RequiredDll "VCOMP140.DLL" @($ompDir) $stageDir | Out-Null

    if ($Kind -eq "cuda" -and -not $SkipCudaRuntimeDlls) {
        $cudaRoot = Find-CudaRoot
        if ($cudaRoot -eq "") {
            throw "CUDA Toolkit was not found, so CUDA DLLs cannot be bundled."
        }
        $cudaDirs = @(
            (Join-Path $cudaRoot "bin\x64"),
            (Join-Path $cudaRoot "bin")
        )
        Copy-RequiredDll "cublas64_13.dll" $cudaDirs $stageDir | Out-Null
        Copy-RequiredDll "cublasLt64_13.dll" $cudaDirs $stageDir | Out-Null
        Copy-RequiredDll "cufft64_12.dll" $cudaDirs $stageDir | Out-Null
    }

    Write-PackageReadme (Join-Path $stageDir "README.md") $Kind $profileSettings (-not ($Kind -eq "cuda" -and $SkipCudaRuntimeDlls))

    $zipPath = Join-Path $OutputDir "$packageName.zip"
    if (Test-Path -LiteralPath $zipPath) {
        Remove-Item -LiteralPath $zipPath -Force
    }
    Compress-Archive -Path (Join-Path $stageDir "*") -DestinationPath $zipPath -Force

    $stageSize = (Get-ChildItem -LiteralPath $stageDir -File -Recurse | Measure-Object Length -Sum).Sum
    $zipSize = (Get-Item -LiteralPath $zipPath).Length
    [pscustomobject]@{
        Package = $packageName
        Directory = $stageDir
        Zip = $zipPath
        UncompressedMB = [math]::Round($stageSize / 1MB, 2)
        ZipMB = [math]::Round($zipSize / 1MB, 2)
    }
}

$repoRoot = Split-Path $PSScriptRoot -Parent
$buildScript = Join-Path $PSScriptRoot "build_windows.ps1"
if ($OutputDir -eq "") {
    $OutputDir = Join-Path (Join-Path $repoRoot "build") "prebuilt"
}
New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null

$effectiveJobs = if ($Jobs -gt 0) { $Jobs } else { [Math]::Max(2, [Environment]::ProcessorCount) }
$Jobs = $effectiveJobs
$profileSettings = Get-ProfileSettings $Profile

$results = @()
if ($Package -eq "all" -or $Package -eq "cpu") {
    $results += New-PrebuiltPackage "windows-cpu-release" "cpu"
}
if ($Package -eq "all" -or $Package -eq "cuda") {
    $results += New-PrebuiltPackage "windows-cuda-release" "cuda"
}

$results | Format-Table -AutoSize
