param(
    [Parameter(Mandatory = $true)]
    [string]$Model,
    [string]$BuildDir = "build/windows-cuda-release",
    [string]$Label = (Get-Date -Format "yyyyMMdd-HHmmss"),
    [string]$Baseline = "",
    [switch]$RequireSameFrames,
    [int]$Device = 0,
    [int]$Threads = 8,
    [int]$Warmup = 1,
    [int]$Iterations = 1
)

$ErrorActionPreference = "Stop"
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "../..")).Path
$ModelPath = (Resolve-Path $Model).Path
$Bench = Join-Path $RepoRoot "$BuildDir/bin/higgs_audio_tts_warm_bench.exe"
$Cases = Join-Path $PSScriptRoot "higgs_audio_tts_cuda_perf_cases.json"
$ResultDir = Join-Path $PSScriptRoot "results/$Label"

if (-not (Test-Path -LiteralPath $Bench)) {
    throw "Warmbench binary does not exist: $Bench"
}

New-Item -ItemType Directory -Force (Join-Path $ResultDir "audio") | Out-Null
$PreviousErrorActionPreference = $ErrorActionPreference
$ErrorActionPreference = "Continue"
& $Bench `
    --model $ModelPath `
    --backend cuda `
    --device $Device `
    --threads $Threads `
    --warmup $Warmup `
    --iterations $Iterations `
    --request-sequence-file $Cases `
    --output-dir (Join-Path $ResultDir "audio") `
    --timing-file (Join-Path $ResultDir "timing.log") 2>&1 |
    ForEach-Object {
        if ($_ -is [System.Management.Automation.ErrorRecord]) {
            $_.Exception.Message
        } else {
            $_
        }
    } |
    Tee-Object -FilePath (Join-Path $ResultDir "console.log")
$BenchExitCode = $LASTEXITCODE
$ErrorActionPreference = $PreviousErrorActionPreference
if ($BenchExitCode -ne 0) {
    exit $BenchExitCode
}

if ($Baseline) {
    $BaselinePath = (Resolve-Path $Baseline).Path
    $CompareArgs = @(
        (Join-Path $PSScriptRoot "compare_warmbench_results.py"),
        "--baseline", $BaselinePath,
        "--candidate", $ResultDir,
        "--output", (Join-Path $ResultDir "comparison.json")
    )
    if ($RequireSameFrames) {
        $CompareArgs += "--require-same-frames"
    }
    python @CompareArgs
    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }
}

Write-Host "Higgs CUDA performance artifacts: $ResultDir"
