$ErrorActionPreference = 'Stop'
if ($env:AIEDU_PYTHON) {
    $analysisPython = $env:AIEDU_PYTHON
} else {
    $analysisPython = (Get-Command python -ErrorAction Stop).Source
}
$analysisEnv = Split-Path -Parent $analysisPython
$env:PATH = "$analysisEnv;$analysisEnv\Library\bin;$analysisEnv\Scripts;$env:PATH"
& $analysisPython @args
exit $LASTEXITCODE
