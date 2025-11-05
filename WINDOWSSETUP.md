# Windows 11 Installation Guide

This guide is specifically for setting up the Family Photo Identifier on your Windows 11 Professional system with RTX 4090.

## System Confirmed
✅ Windows 11 Professional
✅ 256 GB RAM
✅ NVIDIA GeForce RTX 4090 (16 GB VRAM)
✅ Python 3.13

## Quick Installation (5 minutes)

### Option 1: Automated Setup (Recommended)

1. **Extract the project** to your desired location, e.g.:
   ```
   C:\Users\YourName\Documents\family-photo-identifier
   ```

2. **Run the setup script**:
   - Double-click `setup_windows.bat`
   - OR open Command Prompt in the project folder and run:
     ```cmd
     setup_windows.bat
     ```

3. **Wait for installation** (5-10 minutes)
   - Creates virtual environment
   - Installs all dependencies
   - Sets up directory structure

4. **Done!** Follow the on-screen instructions.

### Option 2: Manual Setup

If the automated script doesn't work, follow these steps:

1. **Open PowerShell or Command Prompt** as Administrator

2. **Navigate to project**:
   ```cmd
   cd C:\Users\YourName\Documents\family-photo-identifier
   ```

3. **Create virtual environment**:
   ```cmd
   python -m venv venv
   ```

4. **Activate virtual environment**:
   ```cmd
   venv\Scripts\activate
   ```
   You should see `(venv)` in your prompt.

5. **Install dependencies**:
   ```cmd
   pip install -r requirements.txt
   ```
   This will take 5-10 minutes.

6. **Create .env file**:
   ```cmd
   copy .env.example .env
   ```

7. **Test installation**:
   ```cmd
   python -c "import torch; print('GPU:', torch.cuda.is_available())"
   ```
   Should show `GPU: True`

## GPU Setup (If needed)

If GPU is not detected:

1. **Check NVIDIA Driver**:
   - Press Win + X → Device Manager
   - Expand "Display adapters"
   - Should see "NVIDIA GeForce RTX 4090"
   - If not, install drivers from: https://www.nvidia.com/drivers

2. **Install CUDA Toolkit** (if not present):
   - Download from: https://developer.nvidia.com/cuda-downloads
   - Choose: Windows → x86_64 → 11 → exe (local)
   - Install with default settings

3. **Reinstall PyTorch with CUDA**:
   ```cmd
   pip uninstall torch torchvision
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
   ```

## Ollama Setup (Optional but Recommended)

For enhanced identification with LLaVA:

1. **Download Ollama**:
   - Visit: https://ollama.ai
   - Click "Download for Windows"
   - Run the installer

2. **Install LLaVA model**:
   ```cmd
   ollama pull llava:13b
   ```
   Downloads ~8GB, may take 10-20 minutes

3. **Test Ollama**:
   ```cmd
   ollama run llava:13b
   ```
   Type a test message, then `/bye` to exit

## Usage on Windows

### Activate Environment
Always activate the virtual environment before using the system:
```cmd
cd C:\Users\YourName\Documents\family-photo-identifier
venv\Scripts\activate
```

### Add Training Photos
Copy your photos to the training folder:
```cmd
xcopy "C:\Users\YourName\Pictures\Family\*" data\training\ /E /I
```

### Label Faces
```cmd
python -m src.data_labeler
```

### Train Model
```cmd
python train_model.py
```

### Test Model
```cmd
python test_model.py --image "C:\Users\YourName\Pictures\test_photo.jpg" --display
```

### Process Entire Folder
```cmd
python test_model.py --directory "C:\Users\YourName\Pictures\Family" --output "C:\Users\YourName\Pictures\Family_Identified"
```

## Windows-Specific Tips

### File Paths
Use quotes for paths with spaces:
```cmd
python test_model.py --image "C:\Users\Your Name\Pictures\photo.jpg"
```

### PowerShell vs Command Prompt
Both work, but Command Prompt is simpler for this project.

### HEIC Files
If you have iPhone photos (HEIC format):
```cmd
pip install pillow-heif
```

### Antivirus
Some antivirus may flag Python scripts. Add exception for:
- Project folder
- Python installation folder
- venv folder

### Performance
With your RTX 4090:
- Training: 2-3 minutes for 50 photos
- Face detection: <0.1s per image
- Batch processing: ~1 second per image

## Troubleshooting

### "pip is not recognized"
Python not in PATH. Reinstall Python and check "Add to PATH" option.

### "ImportError: DLL load failed"
Missing Visual C++ Redistributable:
- Download from: https://aka.ms/vs/17/release/vc_redist.x64.exe
- Install and restart

### "CUDA out of memory"
Edit `config.py`:
```python
BATCH_SIZE = 16  # Reduce from 32
```

### Python not found
Install Python 3.13:
1. Visit: https://www.python.org/downloads/
2. Download Python 3.13.x
3. Run installer
4. ✅ Check "Add Python to PATH"
5. Click "Install Now"

### Virtual environment activation fails
Use full path:
```cmd
C:\Users\YourName\Documents\family-photo-identifier\venv\Scripts\activate.bat
```

## Quick Commands Reference

```cmd
# Activate environment
venv\Scripts\activate

# Deactivate environment
deactivate

# Install package
pip install package-name

# Update all packages
pip install --upgrade -r requirements.txt

# Check GPU
python -c "import torch; print(torch.cuda.is_available())"

# Check installed packages
pip list

# View logs
type logs\family_photo_identifier.log

# Clear cache (if issues)
rmdir /s /q __pycache__
rmdir /s /q src\__pycache__
```

## Directory Structure on Windows

```
C:\Users\YourName\Documents\family-photo-identifier\
│
├── venv\                          # Virtual environment (created by setup)
│   ├── Scripts\
│   │   ├── activate.bat          # Activation script
│   │   └── python.exe            # Python in venv
│   └── Lib\                      # Installed packages
│
├── data\                         
│   ├── training\                 # YOUR PHOTOS GO HERE
│   ├── testing\                  # Test photos
│   └── models\                   # Trained models (auto-created)
│
├── src\                          # Source code
├── logs\                         # Log files
│
├── setup_windows.bat             # Setup script (run first)
├── train_model.py                # Training script
├── test_model.py                 # Testing script
├── README.md                     # Full documentation
└── QUICKSTART.md                # Quick start guide
```

## Performance Optimization for Your System

Given your high-end specs (RTX 4090, 256GB RAM):

1. **Edit config.py**:
   ```python
   BATCH_SIZE = 64           # Increase for faster processing
   IMAGE_SIZE = 224          # Higher resolution
   EMBEDDING_MODEL = "ArcFace"  # Best accuracy
   FACE_DETECTION_BACKEND = "retinaface"  # Most accurate
   ```

2. **Enable all CUDA optimizations**:
   ```python
   import torch
   torch.backends.cudnn.benchmark = True
   ```

3. **Process large batches**:
   ```cmd
   python test_model.py --directory "C:\BigPhotoFolder" --output "C:\Results"
   ```
   Your system can easily handle 1000+ photos.

## Next Steps

1. ✅ Run `setup_windows.bat`
2. ✅ Copy training photos to `data\training\`
3. ✅ Run `python -m src.data_labeler`
4. ✅ Run `python train_model.py`
5. ✅ Run `python test_model.py --test`
6. ✅ Process your photo library!

## Support

If you encounter issues:

1. Check `logs\family_photo_identifier.log`
2. Ensure virtual environment is activated
3. Try reinstalling dependencies
4. Check README.md for detailed troubleshooting

---

**Your System is Perfect for This Project!**
The RTX 4090 will make training and inference extremely fast.

Happy photo identifying! 📸