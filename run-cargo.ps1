$ErrorActionPreference = 'Stop'

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = Split-Path -Parent $root

$cargoHome = Join-Path $repoRoot '.tools\cargo'
$rustupHome = Join-Path $repoRoot '.tools\rustup'
$llvmMingw = Join-Path $repoRoot '.tools\llvm-mingw\bin'

if (-not (Test-Path (Join-Path $cargoHome 'bin\cargo.exe'))) {
  throw 'cargo.exe not found. Expected local Rust under .tools\cargo.'
}
if (-not (Test-Path (Join-Path $llvmMingw 'x86_64-w64-mingw32-clang.exe'))) {
  throw 'llvm-mingw linker not found. Expected it under .tools\llvm-mingw\bin.'
}

$env:CARGO_HOME = $cargoHome
$env:RUSTUP_HOME = $rustupHome
$env:PATH = "$($cargoHome)\bin;$llvmMingw;$env:PATH"

& (Join-Path $cargoHome 'bin\cargo.exe') @args
