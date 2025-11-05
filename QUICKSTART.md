# Quick Start Guide

Get started with Family Photo Identifier in 5 simple steps!

## Prerequisites
- Python 3.13 installed
- 50+ family photos ready
- 10-15 minutes of time

## Step 1: Setup (2 minutes)

```bash
# Clone or download the project
cd family-photo-identifier

# Run setup script
python setup.py
```

The setup script will:
- Check Python version
- Create necessary directories
- Install dependencies
- Set up configuration

## Step 2: Add Training Photos (1 minute)

Copy your family photos to the training directory:

```bash
# Windows
copy "C:\Users\YourName\Pictures\Family\*.jpg" data\training\

# Linux/Mac
cp ~/Pictures/Family/*.jpg data/training/
```

**Tips:**
- Include photos with multiple people (it's fine!)
- Mix of lighting conditions and angles is good
- 50+ photos total, aim for 2-5 photos per person minimum

## Step 3: Label Faces (5-10 minutes)

```bash
python -m src.data_labeler
```

Choose option 1 (Console interface), then:
1. View each detected face
2. Type the person's name
3. Press ENTER if unsure
4. Type 'quit' when done

The tool auto-saves progress, so you can exit and resume anytime.

## Step 4: Train Model (2-5 minutes)

```bash
python train_model.py
```

You'll see:
- Training progress
- Number of faces per person
- Final accuracy score

**Goal:** Aim for 75%+ accuracy (90%+ is excellent!)

## Step 5: Test & Use (1 minute)

Test on a single photo:
```bash
python test_model.py --image path/to/photo.jpg --display
```

Process entire directory:
```bash
python test_model.py --directory "C:\Users\YourName\Pictures\More_Family_Photos" --output results
```

---

## Common First-Time Issues

### "No faces detected"
- Check image quality and size
- Ensure faces are at least 50x50 pixels
- Try different detection backend: Edit `.env` and set `FACE_DETECTION_BACKEND=retinaface`

### "Low accuracy (<75%)"
- Add more training photos (aim for 5-10 per person)
- Ensure consistent name spelling during labeling
- Include variety in lighting and angles
- Try different embedding model: `python train_model.py --model ArcFace`

### "GPU not detected"
- System will work on CPU (just slower)
- To enable GPU: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118`

---

## What's Next?

### Improve Accuracy
1. Add more training photos
2. Label test set and run: `python test_model.py --test`
3. Try different models: `python train_model.py --model ArcFace`

### Organize Photo Library
```bash
python test_model.py --directory "C:\Users\YourName\Pictures" --output "C:\Users\YourName\Pictures\Identified"
```

### Optional: Add LLM Verification
1. Install Ollama: https://ollama.ai
2. Pull LLaVA: `ollama pull llava:13b`
3. Use enhanced identification in your scripts

---

## Quick Reference Commands

```bash
# Label training data
python -m src.data_labeler

# Train with defaults
python train_model.py

# Train with options
python train_model.py --model ArcFace --classifier svm

# Test single image
python test_model.py --image photo.jpg --display

# Test accuracy
python test_model.py --test

# Process directory
python test_model.py --directory input_folder --output results_folder

# Check system
python -c "import torch; print(f'GPU: {torch.cuda.is_available()}')"
```

---

## Getting Help

1. Check README.md for detailed instructions
2. Review logs in `logs/family_photo_identifier.log`
3. Ensure all dependencies installed: `pip list | grep -E "(deepface|opencv|torch)"`

---

**You're all set! Happy identifying! 📸**