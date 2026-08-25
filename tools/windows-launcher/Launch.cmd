@echo off
setlocal
pushd "%~dp0..\.."
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0PeoplesLLM-Launcher.ps1"
set "launcher_rc=%errorlevel%"
popd
if not "%launcher_rc%"=="0" pause
exit /b %launcher_rc%
