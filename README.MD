# Family Photo Identifier

An AI-powered system to train a face recognition model on your personal family photos and automatically identify family members in new images.

## Features

- 🎯 **Multi-person detection**: Automatically detects all faces in photos with multiple people
- 🏷️ **Interactive labeling**: Easy-to-use tool for tagging faces with names
- 🧠 **Advanced recognition**: Uses state-of-the-art face embeddings (Facenet512, ArcFace, etc.)
- 📊 **High accuracy**: Achieves 75-90%+ accuracy with proper training data
- 🖼️ **Multi-format support**: Works with JPEG, BMP, HEIC, and GIF images
- 🚀 **GPU accelerated**: Optimized for NVIDIA GPUs
- 🤖 **Optional LLM integration**: Use Ollama or OpenAI for additional verification

## System Requirements

- Windows 11 Professional (or other OS with Python 3.13)
- 256 GB RAM
- NVIDIA GeForce RTX with 16 GB VRAM
- Python 3.13
- Ollama (optional, for LLaVA integration)

## Recommended Model

**Best Open Source LLM for Image Identification: LLaVA 13B**

For your use case, I recommend using the **traditional face recognition approach** (Facenet512 embeddings + SVM classifier) rather than fine-tuning a large vision LLM. Here's why:

### Why Face Recognition Embeddings?

1. **Data efficiency**: Works well with 2-3 photos per person (50 photos for 16-20 people)
2. **Fast training**: Minutes vs hours for LLM fine-tuning
3. **High accuracy**: 75-90% accuracy easily achievable
4. **Resource efficient**: Much less GPU memory than fine-tuning vision LLMs
5. **Proven technology**: Battle-tested for face identification

### Why LLaVA for Optional Verification?

LLaVA (Large Language and Vision Assistant) can be used optionally for:
- Providing context about photos
- Verifying ambiguous identifications
- Counting people in photos
- Scene description

**Model: `llava:13b` via Ollama**

## Installation

### Step 1: Create Virtual Environment

```bash
# Create virtual environment with Python 3.13
python3.13 -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On Linux/Mac:
source venv/bin/activate
```

### Step 2: Install Dependencies

```bash
# Install all required packages
pip install -r requirements.txt
```

### Step 3: Configure Environment

```bash
# Copy environment template
cp .env.example .env

# Edit .env file and set your configuration:
# - OPENAI_API_KEY (optional, only if using OpenAI)
# - OLLAMA_MODEL (default: llava:13b)
# - CONFIDENCE_THRESHOLD (default: 0.75)
```

### Step 4: Install Ollama (Optional)

If you want to use LLaVA for verification:

1. Download Ollama from: https://ollama.ai
2. Install Ollama
3. Pull the LLaVA model:
   ```bash
   ollama pull llava:13b
   ```

## Project Structure

```
family-photo-identifier/
├── config.py                 # Configuration settings
├── requirements.txt          # Python dependencies
├── .env.example             # Environment variables template
├── train_model.py           # Training script
├── test_model.py            # Testing/inference script
├── data/
│   ├── training/            # Put your training photos here
│   ├── testing/             # Put your test photos here
│   └── models/              # Trained models saved here
├── src/
│   ├── utils.py             # Utility functions
│   ├── face_detection.py   # Face detection module
│   ├── face_recognition.py # Face recognition module
│   ├── data_labeler.py     # Interactive labeling tool
│   └── llm_integration.py  # LLM verification (optional)
└── logs/                    # Log files
```

## Usage Guide

### Step 1: Prepare Training Data

1. Create the training directory structure:
   ```bash
   mkdir -p data/training
   ```

2. Copy your family photos to `data/training/`
   - You can organize them in subdirectories or keep them flat
   - Supported formats: .jpg, .jpeg, .png, .bmp, .heic, .gif
   - 50+ photos total with 16-20 people is perfect

### Step 2: Label Faces

Run the interactive labeling tool:

```bash
python -m src.data_labeler
```

**Console Interface (Recommended)**
- Choose option 1 for console interface
- For each detected face, you'll see:
  - Face location and size
  - Detection confidence
- Enter the person's name
- Press ENTER to skip faces you can't identify
- Type 'unknown' for unidentifiable faces
- Type 'quit' to exit and save

**Visual Interface (Alternative)**
- Choose option 2 for visual interface
- Each face will be displayed in a window
- Enter names in the console
- Press any key in the window to continue

**Tips for Labeling:**
- Multiple people per photo is fine - each face will be labeled separately
- Be consistent with names (e.g., always "John" not sometimes "John Smith")
- Skip blurry or unclear faces
- You can run the labeler multiple times - it saves progress

### Step 3: Train the Model

```bash
# Basic training with default settings (Facenet512, SVM classifier)
python train_model.py

# Advanced options:
python train_model.py --model ArcFace --classifier svm --test-split 0.2

# With cross-validation:
python train_model.py --cross-validate
```

**Training Options:**
- `--model`: Embedding model (Facenet512, ArcFace, VGG-Face, OpenFace, DeepFace)
- `--classifier`: Classifier type (svm, knn, random_forest)
- `--test-split`: Fraction for testing (default: 0.2)
- `--cross-validate`: Perform k-fold cross-validation

**Expected Training Time:**
- With your GPU: 2-5 minutes for 50 photos
- CPU only: 10-15 minutes

**Target Accuracy:**
- Minimum: 75% (acceptable)
- Target: 90%+ (excellent)

### Step 4: Test the Model

#### Test on New Images

```bash
# Test single image
python test_model.py --image path/to/photo.jpg --display

# Process entire directory
python test_model.py --directory data/testing --output data/results

# Adjust confidence threshold
python test_model.py --directory data/testing --confidence 0.80
```

#### Evaluate Accuracy

To properly test accuracy, you need labeled test data:

1. Create test directory and add photos:
   ```bash
   mkdir -p data/testing
   cp some_photos/* data/testing/
   ```

2. Label test photos (same as training):
   ```bash
   # Temporarily change to testing directory
   # Edit config.py or use the labeler on testing directory
   python -m src.data_labeler
   ```

3. Run accuracy test:
   ```bash
   python test_model.py --test
   ```

This will show:
- Overall accuracy percentage
- Per-person accuracy breakdown
- Whether you meet the 90% target

### Step 5: Process Photos in Batch

Once trained, process your entire photo library:

```bash
python test_model.py --directory "C:/Users/YourName/Pictures/Family" \
                     --output "C:/Users/YourName/Pictures/Family_Identified" \
                     --confidence 0.75
```

This will:
- Process all photos in the directory
- Save annotated images with names and confidence scores
- Create a JSON summary of all identifications

### Optional: LLM Verification

For enhanced accuracy with LLM verification:

```python
from src.llm_integration import EnhancedIdentifier
from test_model import FaceIdentifier

# Initialize
face_identifier = FaceIdentifier()
enhanced = EnhancedIdentifier(face_identifier, llm_provider='ollama')

# Identify with verification
result = enhanced.identify_with_verification('photo.jpg')

print("Face Recognition:", result['face_recognition'])
print("LLM Verification:", result['llm_verification']['verification'])
```

## Configuration Options

Edit `config.py` or `.env` to customize:

### Face Detection
- `FACE_DETECTION_BACKEND`: mtcnn, retinaface, opencv, ssd, dlib (default: mtcnn)
- `MIN_FACE_SIZE`: Minimum face size in pixels (default: 20)

### Face Recognition
- `EMBEDDING_MODEL`: Facenet512, ArcFace, VGG-Face, etc. (default: Facenet512)
- `CONFIDENCE_THRESHOLD`: Minimum confidence for identification (default: 0.75)

### LLM Integration
- `OLLAMA_HOST`: Ollama server URL (default: http://localhost:11434)
- `OLLAMA_MODEL`: Model name (default: llava:13b)
- `OPENAI_API_KEY`: Your OpenAI API key (optional)

## Troubleshooting

### Low Accuracy (<75%)

1. **Add more training data**: Try to get 5-10 photos per person
2. **Improve photo quality**: Use photos with clear, front-facing faces
3. **Try different models**: Test ArcFace or VGG-Face
4. **Adjust confidence**: Lower threshold to 0.60-0.70
5. **Check labeling**: Ensure names are consistent

### HEIC Images Not Loading

```bash
# Install additional dependencies
pip install pillow-heif
```

### GPU Not Detected

```bash
# Check CUDA installation
python -c "import torch; print(torch.cuda.is_available())"

# Reinstall PyTorch with CUDA support
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Ollama Connection Failed

1. Ensure Ollama is running:
   ```bash
   ollama serve
   ```

2. Test connection:
   ```bash
   python -m src.llm_integration
   ```

### Memory Issues

If you run out of GPU memory:
1. Reduce batch size in config: `BATCH_SIZE = 16`
2. Process images one at a time
3. Use CPU for inference: Set `CUDA_VISIBLE_DEVICES=-1`

## Performance Tips

### For Your RTX 4090 (16GB VRAM):

1. **Optimal Settings:**
   ```python
   BATCH_SIZE = 32
   IMAGE_SIZE = 160
   EMBEDDING_MODEL = "Facenet512"  # Fast and accurate
   ```

2. **Speed Optimizations:**
   - Use MTCNN for face detection (fastest)
   - Process images in batches
   - Keep images under 4K resolution

3. **Accuracy Optimizations:**
   - Use RetinaFace for detection (more accurate but slower)
   - Use ArcFace embeddings (best accuracy)
   - Ensemble multiple models

## Best Practices

### Training Data

1. **Variety**: Include photos with different:
   - Lighting conditions (indoor, outdoor, flash)
   - Angles (front, side profile, slight turn)
   - Expressions (smiling, serious, laughing)
   - Ages (if you have old photos)

2. **Quality**:
   - Clear, in-focus faces
   - Faces at least 50x50 pixels
   - Good lighting
   - Minimal occlusion (no sunglasses, hats)

3. **Labeling**:
   - Use consistent names
   - Skip very blurry faces
   - Label partial faces if the person is identifiable

### Confidence Thresholds

- **0.90+**: Very high confidence - use for critical applications
- **0.75-0.89**: Good confidence - default setting
- **0.60-0.74**: Moderate confidence - review uncertain identifications
- **<0.60**: Low confidence - likely incorrect

## Advanced Features

### Custom Classifier Training

```python
from train_model import ModelTrainer

trainer = ModelTrainer(model_name='ArcFace')
results = trainer.train(classifier_type='svm', test_split=0.2)

# Try ensemble approach
from sklearn.ensemble import VotingClassifier
# Combine multiple classifiers for better accuracy
```

### Face Clustering

```python
# Group similar faces together (for organizing unlabeled photos)
from sklearn.cluster import DBSCAN

# Extract embeddings
embeddings = [...] # Your face embeddings

# Cluster
clustering = DBSCAN(eps=0.5, min_samples=2).fit(embeddings)
```

### Export Results

```python
# Export to CSV for further analysis
import pandas as pd

results = identifier.process_directory('data/testing')
df = pd.DataFrame(results)
df.to_csv('face_identifications.csv', index=False)
```

## FAQ

**Q: Do I need to crop faces manually?**
A: No! The system automatically detects and crops faces from your photos.

**Q: Can it identify people in group photos?**
A: Yes! It detects and identifies all faces in multi-person photos.

**Q: How many photos do I need per person?**
A: Minimum 2-3, recommended 5-10 for best accuracy.

**Q: What if someone looks different in old vs recent photos?**
A: Include photos from different time periods in your training data.

**Q: Can I add more people after training?**
A: Yes! Add their labeled photos and retrain the model.

**Q: Does it work with old photos?**
A: Yes, but quality matters. Scanned photos should be clear and well-lit.

## License

This project is for personal use. Ensure you have rights to all photos you process.

## Support

For issues or questions:
1. Check the troubleshooting section
2. Review log files in `logs/` directory
3. Ensure all dependencies are installed correctly

## Acknowledgments

Built with:
- DeepFace: Face detection and recognition framework
- MTCNN: Face detection
- Facenet/ArcFace: Face embeddings
- scikit-learn: Machine learning classifiers
- LLaVA: Vision-language model (optional)

---

**Note**: This system is designed for personal family photos. Performance may vary based on photo quality, lighting, and diversity of training data.