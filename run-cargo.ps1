$ErrorActionPreference = 'Stop'

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = Split-Path -Parent $root

$cargoHome = Join-Path $repoRoot '.tools\cargo'
$rustupHome = Join-Path $repoRoot '.tools\rustup'
$llvmMingw = Join-Path $repoRoot '.tools\llvm-mingw\bin'
$localCargo = Join-Path $cargoHome 'bin\cargo.exe'
$localClang = Join-Path $llvmMingw 'x86_64-w64-mingw32-clang.exe'

$exitCode = 0
Push-Location $root
try {
  if ((Test-Path $localCargo) -and (Test-Path $localClang)) {
    $env:CARGO_HOME = $cargoHome
    $env:RUSTUP_HOME = $rustupHome
    $env:PATH = "$($cargoHome)\bin;$llvmMingw;$env:PATH"

    & $localCargo @args
    $exitCode = $LASTEXITCODE
  }
  else {
    $cargoCmd = Get-Command cargo -ErrorAction SilentlyContinue
    if (-not $cargoCmd) {
      throw 'cargo executable not found. Install Rust or place portable toolchain under .tools\cargo.'
    }

    if (Test-Path $llvmMingw) {
      $env:PATH = "$llvmMingw;$env:PATH"
    }

    & $cargoCmd.Source @args
    $exitCode = $LASTEXITCODE
  }
}
finally {
  Pop-Location
}

exit $exitCode
