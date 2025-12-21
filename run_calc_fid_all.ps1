# Usage: Run this script from the repo root (Mapperatorinator)
# It will find all subdirectories under ./logs_fid that contain a direct child folder named 'generated'
# For each such directory, it will run: python ./calc_fid.py -cn calc_fid hydra.run.dir=<dir>

param(
    [string]$LogsRoot = "./logs_fid/sweeps",
    [string]$PythonExe = "python"
)

# Resolve to absolute path for safety
$repoRoot = (Get-Location).Path
$logsPath = Resolve-Path -LiteralPath $LogsRoot

Write-Host "Scanning for valid run directories under: $logsPath (recursive)" -ForegroundColor Cyan

# Recursively enumerate directories under logs_fid
$dirs = Get-ChildItem -LiteralPath $logsPath -Directory -Recurse

# Filter directories that have a direct child folder named 'generated'
$validDirs = @()
foreach ($d in $dirs) {
    $generatedPath = Join-Path -Path $d.FullName -ChildPath "generated"
    if (Test-Path -LiteralPath $generatedPath -PathType Container) {
        $validDirs += $d.FullName
    }
}

if ($validDirs.Count -eq 0) {
    Write-Host "No valid directories found (no dir with direct child 'generated')." -ForegroundColor Yellow
    exit 0
}

Write-Host "Found $($validDirs.Count) valid directories:" -ForegroundColor Green
$validDirs | ForEach-Object { Write-Host " - $_" }

# Run calc_fid.py for each directory by overriding hydra.run.dir
$failures = @()
foreach ($runDir in $validDirs) {
    Write-Host "\nRunning calc_fid for: $runDir" -ForegroundColor Cyan

    # Quote the path for Hydra override to handle spaces
    $override = "`"hydra={run:{dir:`'$runDir`'}}`""
    $cmd = "$PythonExe ./calc_fid.py -cn calc_fid $override"

    Write-Host "Command: $cmd" -ForegroundColor DarkGray
    $process = Start-Process -FilePath $PythonExe -ArgumentList "./calc_fid.py","-cn","calc_fid",$override -NoNewWindow -PassThru -Wait

    if ($process.ExitCode -ne 0) {
        Write-Host "calc_fid failed for: $runDir (exit $($process.ExitCode))" -ForegroundColor Red
        $failures += $runDir
    } else {
        Write-Host "calc_fid succeeded for: $runDir" -ForegroundColor Green
    }
}

if ($failures.Count -gt 0) {
    Write-Host "\nCompleted with failures on $($failures.Count) directories:" -ForegroundColor Yellow
    $failures | ForEach-Object { Write-Host " - $_" }
    exit 1
}

Write-Host "\nAll runs completed successfully." -ForegroundColor Green
exit 0

