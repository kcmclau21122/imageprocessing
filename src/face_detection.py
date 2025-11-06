"""
Face detection module using multiple backends
Supports DeepFace with various detectors (MTCNN, RetinaFace, etc.)
Includes automatic image resizing to prevent memory issues with large images
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
    Face detection class supporting multiple backends with automatic
    image resizing to handle large images without memory errors.
    """
    
    def __init__(self, backend: str = None, max_image_size: int = 1920):
        """
        Initialize face detector.
        
        Args:
            backend: Detection backend ('mtcnn', 'retinaface', 'opencv', 'ssd', 'dlib')
            max_image_size: Maximum dimension (width or height) for processing.
                          Images larger than this will be resized to prevent memory issues.
                          Default is 1920 (Full HD). Increase for higher resolution if you have more RAM.
        """
        self.backend = backend or config.FACE_DETECTION_BACKEND
        self.min_face_size = config.MIN_FACE_SIZE
        self.max_image_size = max_image_size
        
        # Initialize MTCNN detector if using that backend
        if self.backend == 'mtcnn':
            self.mtcnn_detector = MTCNN()
            logger.info(f"Initialized MTCNN face detector (max image size: {max_image_size}px)")
        else:
            logger.info(f"Using DeepFace with {self.backend} backend (max image size: {max_image_size}px)")
    
    def _resize_if_needed(self, image: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Resize image if it's too large, maintaining aspect ratio.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Tuple of (resized_image, scale_factor)
            scale_factor is 1.0 if no resizing was needed
        """
        height, width = image.shape[:2]
        max_dimension = max(height, width)
        
        if max_dimension <= self.max_image_size:
            # No resizing needed
            return image, 1.0
        
        # Calculate scale factor
        scale_factor = self.max_image_size / max_dimension
        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)
        
        # Resize image
        resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
        
        logger.info(f"Resized image from {width}x{height} to {new_width}x{new_height} (scale: {scale_factor:.2f})")
        
        return resized, scale_factor
    
    def _scale_boxes(self, faces: List[Dict], scale_factor: float) -> List[Dict]:
        """
        Scale face bounding boxes back to original image coordinates.
        
        Args:
            faces: List of face dictionaries with boxes
            scale_factor: Scale factor used for resizing
            
        Returns:
            List of face dictionaries with scaled boxes
        """
        if scale_factor == 1.0:
            return faces
        
        for face in faces:
            x, y, w, h = face['box']
            
            # Scale back to original coordinates
            face['box'] = [
                int(x / scale_factor),
                int(y / scale_factor),
                int(w / scale_factor),
                int(h / scale_factor)
            ]
        
        return faces
    
    def detect_faces(self, image_path: str) -> List[Dict]:
        """
        Detect faces in an image with automatic resizing for large images.
        
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
            
            # Check image size and log if it's large
            height, width = image.shape[:2]
            if max(height, width) > self.max_image_size:
                logger.info(f"Processing large image: {width}x{height}px from {image_path}")
            
            # Detect faces using selected backend
            if self.backend == 'mtcnn':
                return self._detect_with_mtcnn(image, image_path)
            else:
                return self._detect_with_deepface(image_path)
                
        except MemoryError as e:
            logger.error(f"Memory error processing {image_path}: {str(e)}")
            logger.error(f"Try reducing MAX_IMAGE_SIZE in config or use a different backend")
            return []
        except Exception as e:
            logger.error(f"Error detecting faces in {image_path}: {str(e)}")
            return []
    
    def _detect_with_mtcnn(self, image: np.ndarray, image_path: str) -> List[Dict]:
        """
        Detect faces using MTCNN detector with automatic resizing.
        
        Args:
            image: Input image as numpy array
            image_path: Path to original image (for extracting full-resolution faces)
            
        Returns:
            List of face dictionaries
        """
        faces = []
        
        try:
            # Resize image if needed
            resized_image, scale_factor = self._resize_if_needed(image)
            
            # Detect faces on resized image
            try:
                detections = self.mtcnn_detector.detect_faces(resized_image)
            except MemoryError as e:
                logger.error(f"MTCNN memory error even after resizing. Image may still be too large.")
                logger.error(f"Try reducing max_image_size further or use 'opencv' backend instead")
                return []
            
            for detection in detections:
                # Extract bounding box from resized image
                x, y, w, h = detection['box']
                confidence = detection['confidence']
                
                # Scale box back to original image coordinates
                if scale_factor != 1.0:
                    x_orig = int(x / scale_factor)
                    y_orig = int(y / scale_factor)
                    w_orig = int(w / scale_factor)
                    h_orig = int(h / scale_factor)
                else:
                    x_orig, y_orig, w_orig, h_orig = x, y, w, h
                
                # Filter by minimum size
                if w_orig >= self.min_face_size and h_orig >= self.min_face_size:
                    # Extract face region from ORIGINAL image for best quality
                    # Ensure coordinates are within bounds
                    y_start = max(0, y_orig)
                    y_end = min(image.shape[0], y_orig + h_orig)
                    x_start = max(0, x_orig)
                    x_end = min(image.shape[1], x_orig + w_orig)
                    
                    face_img = image[y_start:y_end, x_start:x_end]
                    
                    faces.append({
                        'box': [x_orig, y_orig, w_orig, h_orig],
                        'confidence': confidence,
                        'face_img': face_img,
                        'keypoints': detection.get('keypoints', {})
                    })
            
            logger.info(f"Detected {len(faces)} faces using MTCNN")
            
        except Exception as e:
            logger.error(f"MTCNN detection error: {str(e)}")
            logger.error(f"Consider switching to 'opencv' or 'retinaface' backend if problems persist")
        
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


def batch_detect_faces(image_paths: List[str], backend: str = None, 
                      max_image_size: int = 1920) -> Dict[str, List[Dict]]:
    """
    Detect faces in multiple images with memory-safe processing.
    
    Args:
        image_paths: List of image file paths
        backend: Detection backend to use
        max_image_size: Maximum image dimension for processing
        
    Returns:
        Dictionary mapping image paths to lists of detected faces
    """
    detector = FaceDetector(backend, max_image_size=max_image_size)
    results = {}
    failed_images = []
    
    from tqdm import tqdm
    
    for image_path in tqdm(image_paths, desc="Detecting faces"):
        try:
            faces = detector.detect_faces(image_path)
            if faces:
                results[image_path] = faces
        except Exception as e:
            logger.error(f"Failed to process {image_path}: {str(e)}")
            failed_images.append(image_path)
    
    total_faces = sum(len(faces) for faces in results.values())
    logger.info(f"Detected {total_faces} total faces in {len(results)} images")
    
    if failed_images:
        logger.warning(f"Failed to process {len(failed_images)} images")
    
    return results