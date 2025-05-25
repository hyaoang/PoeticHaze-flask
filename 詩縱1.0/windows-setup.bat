@echo off
REM Ensure the console uses UTF-8 encoding (best effort)
REM chcp 65001

title Game Server

echo Setting up and starting game server...

REM Change the current directory to the folder where this .bat file is located
cd /d "%~dp0"

REM --- Force delete the existing virtual environment folder ---
echo Removing existing virtual environment (if any)...
rmdir /s /q venv
REM ------------------------------------------------------------

REM Create the virtual environment using the 'python' command found in the system's PATH
echo Creating virtual environment...
REM Since 'python --version' works, we assume 'python' is in PATH and proceed.
python -m venv venv
IF ERRORLEVEL 1 (
    echo Error: Could not create virtual environment. Please make sure Python is installed and added to your system's PATH environment variables.
    echo You can download Python from https://www.python.org/downloads/
    pause
    exit /b 1
)

echo Activating virtual environment...
REM Call the activate script for the virtual environment
call venv\Scripts\activate
IF ERRORLEVEL 1 (
    echo Error: Could not activate virtual environment. Check venv creation or Python installation.
    pause
    exit /b 1
)

REM --- Optional: Check Python version in virtual environment ---
echo Checking Python version in virtual environment:
python --version
REM --------------------------------------------------------

echo Installing or updating required packages (this may take a moment)...
REM Use the pip from the activated virtual environment
pip install -r requirements.txt
IF ERRORLEVEL 1 (
    echo Error: Failed to install packages. Check requirements.txt or network connection.
    pause
    exit /b 1
)

echo Starting game server...
echo Please open your web browser and go to: http://127.0.0.1:5000/
echo Close this window to stop the server.

REM Open the game URL in the default web browser
start "" http://127.0.0.1:5000/

REM Run the Flask application using the python from the activated virtual environment
python app.py

echo Server stopped. Press any key to exit...
pause