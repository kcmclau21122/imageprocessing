"""
Setup script for Family Photo Identifier
Automates the installation and configuration process
"""
import os
import sys
import subprocess
from pathlib import Path


def print_header(text):
    """Print formatted header."""
    print("\n" + "="*60)
    print(text)
    print("="*60 + "\n")


def check_python_version():
    """Check if Python version is 3.13+."""
    print_header("Checking Python Version")
    
    version = sys.version_info
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 10):
        print("❌ Python 3.10 or higher is required")
        print(f"   Current version: {version.major}.{version.minor}")
        return False
    
    print("✅ Python version is compatible")
    return True


def create_directories():
    """Create necessary directories."""
    print_header("Creating Directory Structure")
    
    directories = [
        "data/training",
        "data/testing",
        "data/models",
        "logs",
        "src"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✅ Created: {directory}")


def setup_environment():
    """Create .env file from template."""
    print_header("Setting Up Environment")
    
    env_file = Path(".env")
    env_example = Path(".env.example")
    
    if env_file.exists():
        print("⚠️  .env file already exists")
        response = input("   Overwrite? (y/n): ").strip().lower()
        if response != 'y':
            print("   Skipping .env setup")
            return
    
    if env_example.exists():
        import shutil
        shutil.copy(env_example, env_file)
        print("✅ Created .env file from template")
        print("   Please edit .env to configure your settings")
    else:
        print("⚠️  .env.example not found")


def install_dependencies():
    """Install required Python packages."""
    print_header("Installing Dependencies")
    
    print("This will install all required packages...")
    print("This may take 5-10 minutes depending on your internet connection.\n")
    
    response = input("Continue? (y/n): ").strip().lower()
    if response != 'y':
        print("Skipping dependency installation")
        return False
    
    try:
        print("\nInstalling packages...\n")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
        ])
        print("\n✅ All dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Error installing dependencies: {e}")
        return False


def check_gpu():
    """Check if GPU is available."""
    print_header("Checking GPU Availability")
    
    try:
        import torch
        
        if torch.cuda.is_available():
            print(f"✅ GPU detected: {torch.cuda.get_device_name(0)}")
            print(f"   CUDA version: {torch.version.cuda}")
            print(f"   GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        else:
            print("⚠️  No GPU detected")
            print("   The system will work but may be slower")
            print("   Consider installing CUDA-enabled PyTorch")
    except ImportError:
        print("⚠️  PyTorch not installed yet")


def test_imports():
    """Test if all required packages can be imported."""
    print_header("Testing Package Imports")
    
    packages = [
        ('numpy', 'NumPy'),
        ('cv2', 'OpenCV'),
        ('PIL', 'Pillow'),
        ('deepface', 'DeepFace'),
        ('sklearn', 'scikit-learn'),
        ('torch', 'PyTorch')
    ]
    
    all_success = True
    
    for module_name, display_name in packages:
        try:
            __import__(module_name)
            print(f"✅ {display_name}")
        except ImportError:
            print(f"❌ {display_name} - Not installed")
            all_success = False
    
    return all_success


def setup_ollama():
    """Guide user through Ollama setup."""
    print_header("Ollama Setup (Optional)")
    
    print("Ollama allows you to run LLaVA locally for enhanced face identification.")
    print("This is optional but recommended for best results.\n")
    
    response = input("Would you like instructions to set up Ollama? (y/n): ").strip().lower()
    
    if response == 'y':
        print("\nOllama Setup Instructions:")
        print("1. Download Ollama from: https://ollama.ai")
        print("2. Install Ollama for your operating system")
        print("3. Open a terminal and run: ollama pull llava:13b")
        print("4. Wait for the model to download (~8GB)")
        print("5. Test with: ollama run llava:13b")
        print("\nNote: You can skip this and use the system without LLM verification")


def create_sample_structure():
    """Create sample data structure guide."""
    print_header("Data Organization Guide")
    
    guide_file = Path("data/DATA_ORGANIZATION.txt")
    
    guide_content = """
FAMILY PHOTO IDENTIFIER - DATA ORGANIZATION GUIDE
================================================

TRAINING DATA (data/training/)
-----------------------------
Put all photos you want to use for training here.

Structure Option 1: Flat (Recommended for mixed photos)
data/training/
  ├── family_reunion_2023.jpg
  ├── christmas_2022.jpg
  ├── birthday_party.jpg
  └── vacation_2021.jpg

Structure Option 2: Organized by event
data/training/
  ├── reunion_2023/
  │   ├── group_photo.jpg
  │   └── dinner.jpg
  └── christmas_2022/
      ├── tree_lighting.jpg
      └── gifts.jpg

The labeling tool will find all images regardless of subdirectory structure.

TESTING DATA (data/testing/)
---------------------------
Put photos for testing accuracy here after training.
Use the same structure as training data.

IMPORTANT NOTES:
- Each photo can contain multiple people
- All faces will be detected automatically
- You'll label each detected face individually
- Supported formats: .jpg, .jpeg, .png, .bmp, .heic, .gif
- Minimum recommended: 50+ photos with 16-20 people
- For best results: 5-10 photos per person

WORKFLOW:
1. Copy all training photos to data/training/
2. Run: python -m src.data_labeler
3. Label each detected face with the person's name
4. Run: python train_model.py
5. Test with: python test_model.py --test
"""
    
    with open(guide_file, 'w') as f:
        f.write(guide_content)
    
    print(f"✅ Created data organization guide: {guide_file}")


def print_next_steps():
    """Print next steps for the user."""
    print_header("Setup Complete! Next Steps:")
    
    print("1. Add your training photos:")
    print("   - Copy photos to: data/training/")
    print("   - Include photos with all family members")
    print("   - Aim for 50+ photos with 16-20 people\n")
    
    print("2. Label the faces:")
    print("   python -m src.data_labeler\n")
    
    print("3. Train the model:")
    print("   python train_model.py\n")
    
    print("4. Test the model:")
    print("   python test_model.py --test\n")
    
    print("5. Identify faces in new photos:")
    print("   python test_model.py --directory path/to/photos\n")
    
    print("For detailed instructions, see README.md")
    print("\nGood luck with your family photo identification project! 🎉")


def main():
    """Main setup function."""
    print("\n")
    print("┌─────────────────────────────────────────────────────┐")
    print("│                                                     │")
    print("│        FAMILY PHOTO IDENTIFIER - SETUP              │")
    print("│                                                     │")
    print("└─────────────────────────────────────────────────────┘")
    
    # Check Python version
    if not check_python_version():
        print("\n❌ Setup cannot continue with incompatible Python version")
        sys.exit(1)
    
    # Create directories
    create_directories()
    
    # Setup environment
    setup_environment()
    
    # Install dependencies
    deps_installed = install_dependencies()
    
    if deps_installed:
        # Test imports
        test_imports()
        
        # Check GPU
        check_gpu()
    
    # Create sample structure
    create_sample_structure()
    
    # Ollama setup guide
    setup_ollama()
    
    # Print next steps
    print_next_steps()


if __name__ == "__main__":
    main()