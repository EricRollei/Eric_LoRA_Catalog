@echo off
rem The card editor: http://127.0.0.1:8765 (opens in your browser). Close this window to stop it.
rem Python 3.12: ComfyUI portable's python_embeded when this folder is in its custom_nodes, else the py launcher, else python.
set "HERE=%~dp0"
set "EMB=%HERE%..\..\..\python_embeded\python.exe"
if exist "%EMB%" (
    "%EMB%" -X utf8 "%HERE%run.py" serve %*
) else (
    where py >nul 2>nul && ( py -3.12 -X utf8 "%HERE%run.py" serve %* ) || ( python -X utf8 "%HERE%run.py" serve %* )
)
pause
