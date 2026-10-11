@echo off
rem Brings the catalog up to date after adding LoRAs (every step skips what is done). Log: <data folder>\logs\refresh.log
rem Python 3.12: ComfyUI portable's python_embeded when this folder is in its custom_nodes, else the py launcher, else python.
set "HERE=%~dp0"
set "EMB=%HERE%..\..\..\python_embeded\python.exe"
if exist "%EMB%" (
    "%EMB%" -X utf8 "%HERE%run.py" refresh %*
) else (
    where py >nul 2>nul && ( py -3.12 -X utf8 "%HERE%run.py" refresh %* ) || ( python -X utf8 "%HERE%run.py" refresh %* )
)
if not defined NOPAUSE pause
