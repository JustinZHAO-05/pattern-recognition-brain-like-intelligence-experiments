param(
    [ValidateSet("smoke", "full")]
    [string]$Profile = "full",
    [ValidateSet("setup", "data", "experiments", "figures", "report", "check", "all")]
    [string]$Stage = "all",
    [ValidateSet("auto", "cuda", "cpu")]
    [string]$Device = "auto",
    [int]$Seed = 2026
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$Venv = Join-Path $Root ".venv"
$Python = Join-Path $Venv "Scripts\python.exe"
$RuntimePython = "C:\Users\29542\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"

if (-not (Test-Path $Python)) {
    if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
        throw "uv is required to create the reproducible experiment environment."
    }
    uv venv $Venv --python $RuntimePython
}

$env:TORCH_HOME = (Resolve-Path (Join-Path $Root "models")).Path
$env:PYTHONIOENCODING = "utf-8"

if ($Stage -eq "setup" -or $Stage -eq "all") {
    uv pip install --python $Python torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
    uv pip install --python $Python matplotlib pandas scikit-learn seaborn tqdm pycocotools opencv-python pdfplumber pypdf reportlab
}

& $Python -X utf8 (Join-Path $Root "submission\run_all.py") --profile $Profile --stage $Stage --device $Device --seed $Seed
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}
