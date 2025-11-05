# Family Photo Identifier - PowerShell Setup Script
# Run this script to set up the project on Windows 11
# Uses Python 3.13 from: C:\Users\mclau\AppData\Local\Programs\Python\Python313

Write-Host "===============================================================" -ForegroundColor Cyan
Write-Host "      FAMILY PHOTO IDENTIFIER - WINDOWS SETUP" -ForegroundColor Cyan
Write-Host "===============================================================" -ForegroundColor Cyan
Write-Host ""

# Set Python path
$PythonPath = "C:\Users\mclau\AppData\Local\Programs\Python\Python313\python.exe"

# Check if Python exists at specified path
Write-Host "Checking Python installation..." -ForegroundColor Yellow
if (-not (Test-Path $PythonPath)) {
    Write-Host "ERROR: Python 3.13 not found at: $PythonPath" -ForegroundColor Red
    Write-Host "Please verify the Python installation path" -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

Write-Host "Found Python 3.13 at: $PythonPath" -ForegroundColor Green
& $PythonPath --version
Write-Host ""

# Create virtual environment
Write-Host "Creating virtual environment with Python 3.13..." -ForegroundColor Yellow
& $PythonPath -m venv venv
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Failed to create virtual environment" -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}
Write-Host "Virtual environment created successfully!" -ForegroundColor Green
Write-Host ""

# Activate virtual environment
Write-Host "Activating virtual environment..." -ForegroundColor Yellow
& .\venv\Scripts\Activate.ps1
Write-Host ""

# Upgrade pip
Write-Host "Upgrading pip..." -ForegroundColor Yellow
python -m pip install --upgrade pip
Write-Host ""

# Install dependencies
Write-Host "Installing dependencies..." -ForegroundColor Yellow
Write-Host "This may take 5-10 minutes..." -ForegroundColor Yellow
Write-Host ""
pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Failed to install dependencies" -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}
Write-Host ""

# Create .env file if it doesn't exist
Write-Host "Setting up environment configuration..." -ForegroundColor Yellow
if (-not (Test-Path .env)) {
    if (Test-Path .env.example) {
        Copy-Item .env.example .env
        Write-Host "Created .env file - please edit it to configure your settings" -ForegroundColor Green
    } else {
        Write-Host "Warning: .env.example not found, skipping .env creation" -ForegroundColor Yellow
    }
} else {
    Write-Host ".env file already exists" -ForegroundColor Green
}
Write-Host ""

# Create directories
Write-Host "Creating project directories..." -ForegroundColor Yellow
$directories = @("data\training", "data\testing", "data\models", "logs")
foreach ($dir in $directories) {
    if (-not (Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir -Force | Out-Null
    }
}
Write-Host "Directories created!" -ForegroundColor Green
Write-Host ""

# Check GPU
Write-Host "Checking GPU availability..." -ForegroundColor Yellow
python -c "import torch; print('GPU Available:', torch.cuda.is_available()); print('GPU Name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'); print('CUDA Version:', torch.version.cuda if torch.cuda.is_available() else 'N/A')"
Write-Host ""

Write-Host "===============================================================" -ForegroundColor Cyan
Write-Host "                   SETUP COMPLETE!" -ForegroundColor Green
Write-Host "===============================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Python 3.13 Location: $PythonPath" -ForegroundColor Cyan
Write-Host "Virtual Environment: $PWD\venv" -ForegroundColor Cyan
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "  1. Keep this PowerShell window open (venv is activated)" -ForegroundColor White
Write-Host "  2. Copy your family photos to: data\training\" -ForegroundColor White
Write-Host "  3. Label faces: python -m src.data_labeler" -ForegroundColor White
Write-Host "  4. Train model: python train_model.py" -ForegroundColor White
Write-Host "  5. Test model: python test_model.py --test" -ForegroundColor White
Write-Host ""
Write-Host "To activate venv in future PowerShell sessions:" -ForegroundColor Yellow
Write-Host "  .\venv\Scripts\Activate.ps1" -ForegroundColor Cyan
Write-Host ""
Write-Host "For detailed instructions, see README.md" -ForegroundColor Yellow
Write-Host "For quick start, see QUICKSTART.md" -ForegroundColor Yellow
Write-Host ""
Read-Host "Press Enter to exit"
