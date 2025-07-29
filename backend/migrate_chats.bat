@echo off
REM Script to run the chat migration CLI tool on Windows

cd /d "%~dp0"

REM Check if virtual environment exists and activate it
if exist "venv\Scripts\activate.bat" (
    echo Activating virtual environment...
    call venv\Scripts\activate.bat
) else if exist ".venv\Scripts\activate.bat" (
    echo Activating virtual environment...
    call .venv\Scripts\activate.bat
)

REM Check if Python script exists
if not exist "migrate_chats.py" (
    echo Error: migrate_chats.py not found in %CD%
    exit /b 1
)

REM Run the migration script with all arguments passed through
python migrate_chats.py %* 