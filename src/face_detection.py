"""
Face detection module using multiple backends
Supports DeepFace with various detectors (MTCNN, RetinaFace, etc.)
"""
import logging
from typing import List, Tuple, Dict, Optional
import numpy as np
import cv2
from deepface import DeepFace
from mtcnn import MTCNN

import config
from src.utils import load_image

logger = logging.getLogger(__name__)


class FaceDetector:
    """
    Face detection class supporting multiple backends.
    """
    
    def __init__(self, backend: str = None):
        """
        Initialize face detector.
        
        Args:
            backend: Detection backend ('mtcnn', 'retinaface', 'opencv', 'ssd', 'dlib')
        """
        self.backend = backend or config.FACE_DETECTION_BACKEND
        self.min_face_size = config.MIN_FACE_SIZE
        
        # Initialize MTCNN detector if using that backend
        if self.backend == 'mtcnn':
            self.mtcnn_detector = MTCNN()
            logger.info("Initialized MTCNN face detector")
        else:
            logger.info(f"Using DeepFace with {self.backend} backend")
    
    def detect_faces(self, image_path: str) -> List[Dict]:
        """
        Detect faces in an image.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            List of dictionaries containing face information:
            [{'box': [x, y, w, h], 'confidence': float, 'face_img': np.ndarray}, ...]
        """
        try:
            # Load image
            image = load_image(image_path)
            if image is None:
                logger.error(f"Failed to load image: {image_path}")
                return []
            
            # Detect faces using selected backend
            if self.backend == 'mtcnn':
                return self._detect_with_mtcnn(image)
            else:
                return self._detect_with_deepface(image_path)
                
        except Exception as e:
            logger.error(f"Error detecting faces in {image_path}: {str(e)}")
            return []
    
    def _detect_with_mtcnn(self, image: np.ndarray) -> List[Dict]:
        """
        Detect faces using MTCNN detector.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            List of face dictionaries
        """
        faces = []
        
        try:
            # Detect faces
            detections = self.mtcnn_detector.detect_faces(image)
            
            for detection in detections:
                # Extract bounding box
                x, y, w, h = detection['box']
                confidence = detection['confidence']
                
                # Filter by minimum size and confidence
                if w >= self.min_face_size and h >= self.min_face_size:
                    # Extract face region
                    face_img = image[y:y+h, x:x+w]
                    
                    faces.append({
                        'box': [x, y, w, h],
                        'confidence': confidence,
                        'face_img': face_img,
                        'keypoints': detection.get('keypoints', {})
                    })
            
            logger.info(f"Detected {len(faces)} faces using MTCNN")
            
        except Exception as e:
            logger.error(f"MTCNN detection error: {str(e)}")
        
        return faces
    
    def _detect_with_deepface(self, image_path: str) -> List[Dict]:
        """
        Detect faces using DeepFace with specified backend.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            List of face dictionaries
        """
        faces = []
        
        try:
            # Extract faces using DeepFace
            face_objs = DeepFace.extract_faces(
                img_path=image_path,
                detector_backend=self.backend,
                enforce_detection=False,
                align=True
            )
            
            for face_obj in face_objs:
                # Get facial area
                facial_area = face_obj.get('facial_area', {})
                x = facial_area.get('x', 0)
                y = facial_area.get('y', 0)
                w = facial_area.get('w', 0)
                h = facial_area.get('h', 0)
                
                confidence = face_obj.get('confidence', 0.0)
                
                # Filter by minimum size
                if w >= self.min_face_size and h >= self.min_face_size:
                    # Get face image
                    face_img = face_obj.get('face', None)
                    if face_img is not None:
                        # Convert to uint8 if needed
                        if face_img.dtype != np.uint8:
                            face_img = (face_img * 255).astype(np.uint8)
                        
                        faces.append({
                            'box': [x, y, w, h],
                            'confidence': confidence,
                            'face_img': face_img
                        })
            
            logger.info(f"Detected {len(faces)} faces using DeepFace ({self.backend})")
            
        except Exception as e:
            logger.error(f"DeepFace detection error: {str(e)}")
        
        return faces
    
    def detect_and_extract(self, image_path: str, output_dir: str = None) -> List[Dict]:
        """
        Detect faces and optionally save them to disk.
        
        Args:
            image_path: Path to the image file
            output_dir: Directory to save face crops (optional)
            
        Returns:
            List of face dictionaries with file paths if saved
        """
        faces = self.detect_faces(image_path)
        
        if output_dir and faces:
            from pathlib import Path
            import os
            
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            image_name = Path(image_path).stem
            
            for i, face in enumerate(faces):
                face_path = output_dir / f"{image_name}_face_{i}.jpg"
                
                # Save face image
                face_img = face['face_img']
                face_bgr = cv2.cvtColor(face_img, cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(face_path), face_bgr)
                
                face['file_path'] = str(face_path)
        
        return faces


def batch_detect_faces(image_paths: List[str], backend: str = None) -> Dict[str, List[Dict]]:
    """
    Detect faces in multiple images.
    
    Args:
        image_paths: List of image file paths
        backend: Detection backend to use
        
    Returns:
        Dictionary mapping image paths to lists of detected faces
    """
    detector = FaceDetector(backend)
    results = {}
    
    from tqdm import tqdm
    
    for image_path in tqdm(image_paths, desc="Detecting faces"):
        faces = detector.detect_faces(image_path)
        if faces:
            results[image_path] = faces
    
    total_faces = sum(len(faces) for faces in results.values())
    logger.info(f"Detected {total_faces} total faces in {len(results)} images")
    
    return results