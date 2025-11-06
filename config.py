"""
Configuration file for Family Photo Identifier
Loads environment variables and sets default configurations
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Project paths
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
TRAINING_DIR = DATA_DIR / "training"
TESTING_DIR = DATA_DIR / "testing"
MODELS_DIR = DATA_DIR / "models"
LOGS_DIR = PROJECT_ROOT / "logs"

# Ensure directories exist
for directory in [DATA_DIR, TRAINING_DIR, TESTING_DIR, MODELS_DIR, LOGS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# API Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llava:13b")

# Face Detection Configuration
FACE_DETECTION_BACKEND = os.getenv("FACE_DETECTION_BACKEND", "mtcnn")
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.75"))
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "Facenet512")

# Image Processing Configuration
MIN_FACE_SIZE = int(os.getenv("MIN_FACE_SIZE", "20"))
IMAGE_SIZE = int(os.getenv("IMAGE_SIZE", "160"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "32"))

# Maximum image dimension for face detection to prevent memory errors
# Images larger than this will be resized before processing
# Default: 1920 (Full HD)
# Increase if you have more RAM and want to process higher resolution images
# Decrease if you encounter memory errors (try 1280 or 960)
MAX_IMAGE_SIZE = int(os.getenv("MAX_IMAGE_SIZE", "1920"))

# Supported image formats
SUPPORTED_FORMATS = ['.jpg', '.jpeg', '.png', '.bmp', '.heic', '.gif']

# Model paths
EMBEDDINGS_FILE = MODELS_DIR / "face_embeddings.pkl"
LABELS_FILE = MODELS_DIR / "face_labels.json"
CLASSIFIER_FILE = MODELS_DIR / "face_classifier.pkl"

# Logging configuration
LOG_FILE = LOGS_DIR / "family_photo_identifier.log"
LOG_LEVEL = "INFO"