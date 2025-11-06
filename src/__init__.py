"""
Family Photo Identifier - Source Module
Face detection, recognition, and identification system
"""

__version__ = "1.0.0"
__author__ = "Family Photo Identifier Project"

from .face_detection import FaceDetector, batch_detect_faces
from .face_recognition import FaceRecognizer
from .utils import load_image, get_image_files

__all__ = [
    'FaceDetector',
    'FaceRecognizer',
    'batch_detect_faces',
    'load_image',
    'get_image_files',
]