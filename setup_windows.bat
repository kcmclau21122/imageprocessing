@echo off
REM Family Photo Identifier - Windows Setup Script
REM Run this script to set up the project on Windows 11

echo ===============================================================
echo       FAMILY PHOTO IDENTIFIER - WINDOWS SETUP
echo ===============================================================
echo.

REM Check Python installation
echo Checking Python installation...
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.13 from https://www.python.org
    pause
    exit /b 1
)

python --version
echo.

REM Create virtual environment
echo Creating virtual environment...
python -m venv venv
if errorlevel 1 (
    echo ERROR: Failed to create virtual environment
    pause
    exit /b 1
)
echo Virtual environment created successfully!
echo.

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat
"venv\Scripts\activate.ps1" 
echo.

REM Install dependencies
echo Installing dependencies...
echo This may take 5-10 minutes...
echo.
pip install -r requirements.txt
if errorlevel 1 (
    echo ERROR: Failed to install dependencies
    pause
    exit /b 1
)
echo.

REM Create .env file
echo Setting up environment configuration...
if not exist .env (
    copy .env.example .env
    echo Created .env file - please edit it to configure your settings
) else (
    echo .env file already exists
)
echo.

REM Create directories
echo Creating project directories...
if not exist "data\training" mkdir "data\training"
if not exist "data\testing" mkdir "data\testing"
if not exist "data\models" mkdir "data\models"
if not exist "logs" mkdir "logs"
echo Directories created!
echo.

REM Check GPU
echo Checking GPU availability...
python -c "import torch; print('GPU Available:', torch.cuda.is_available()); print('GPU Name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
echo.

echo ===============================================================
echo                    SETUP COMPLETE!
echo ===============================================================
echo.
echo Next steps:
echo   1. Copy your family photos to: data\training\
echo   2. Activate virtual environment: venv\Scripts\activate.bat
echo   3. Label faces: python -m src.data_labeler
echo   4. Train model: python train_model.py
echo   5. Test model: python test_model.py --test
echo.
echo For detailed instructions, see README.md
echo For quick start, see QUICKSTART.md
echo.
echo Press any key to exit...
pause >nul