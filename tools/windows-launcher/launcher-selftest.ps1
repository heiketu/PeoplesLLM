$ErrorActionPreference = "Stop"
Import-Module (Join-Path $PSScriptRoot "PeoplesLLM.Launcher.psm1") -Force
Write-Output (Invoke-LauncherCoreSelfTest)
$profile = Load-LauncherProfile (Join-Path $PSScriptRoot "sample-profile.json")
$errors = @(Test-LauncherProfile $profile)
if ($errors.Count -ne 0) { throw ($errors -join [Environment]::NewLine) }
$command = ConvertTo-LauncherCommand $profile
if ($command.Arguments -notcontains "draft-dspark") { throw "sample profile did not enable DSpark" }
if ($command.Arguments -notcontains "CUDA0,CUDA1") { throw "sample profile lost the device list" }
Write-Output "PeoplesLLM launcher sample profile selftest: PASS"
