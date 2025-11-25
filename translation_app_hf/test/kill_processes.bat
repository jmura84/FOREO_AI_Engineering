@echo off
rem -------------------------------------------------
rem Force‑kill any process listening on ports 8000‑8002
rem -------------------------------------------------
for %%P in (8000 8001 8002) do (
    echo Checking port %%P ...
    for /f "tokens=5" %%A in ('netstat -ano ^| findstr :%%P ^| findstr LISTENING') do (
        echo Killing PID %%A that is bound to port %%P
        taskkill /F /PID %%A >nul 2>&1
    )
)

rem -------------------------------------------------
rem OPTIONAL: Kill all running python.exe processes
rem -------------------------------------------------
echo.
echo Killing any stray python.exe processes...
rem The /F flag forces termination; /IM matches the image name.
taskkill /F /IM python.exe >nul 2>&1

echo All done.