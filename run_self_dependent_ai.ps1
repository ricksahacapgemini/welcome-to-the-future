param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]] $Args
)

begin {
    $stdinValues = @()
}

process {
    $stdinValues += $_
}

end {
    $ErrorActionPreference = "Stop"

    $projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
    $candidates = @(
        (Join-Path $projectRoot ".venv\Scripts\python.exe"),
        'C:\Users\ricsaha\AppData\Roaming\uv\python\cpython-3.14.8-windows-x86_64-none\python.exe'
    )

    foreach ($candidate in $candidates) {
        if (Test-Path $candidate) {
            if ($stdinValues.Count -gt 0) {
                $stdinValues | & $candidate -m self_dependent_ai.cli @Args
                exit $LASTEXITCODE
            }

            & $candidate -m self_dependent_ai.cli @Args
            exit $LASTEXITCODE
        }
    }

    if (Get-Command py -ErrorAction SilentlyContinue) {
        if ($stdinValues.Count -gt 0) {
            $stdinValues | & py -3 -m self_dependent_ai.cli @Args
            exit $LASTEXITCODE
        }

        & py -3 -m self_dependent_ai.cli @Args
        exit $LASTEXITCODE
    }

    if (Get-Command python -ErrorAction SilentlyContinue) {
        if ($stdinValues.Count -gt 0) {
            $stdinValues | & python -m self_dependent_ai.cli @Args
            exit $LASTEXITCODE
        }

        & python -m self_dependent_ai.cli @Args
        exit $LASTEXITCODE
    }

    Write-Error "Python is not available in PATH. Install Python 3.10+ or activate a venv before running this project."
    exit 1
}
