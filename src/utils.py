"""
Utility functions for image processing and face detection
"""
import os
import logging
from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np
from PIL import Image
import cv2
from pillow_heif import register_heif_opener

# Register HEIF opener for Pillow
register_heif_opener()

from . import config

# Setup logging
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(config.LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def load_image(image_path: str) -> Optional[np.ndarray]:
    """
    Load an image from file, supporting multiple formats including HEIC.
    Handles both absolute and relative paths (relative paths are resolved from PROJECT_ROOT).

    Args:
        image_path: Path to the image file (absolute or relative to PROJECT_ROOT)

    Returns:
        numpy.ndarray: Image in RGB format (or None if loading fails)
    """
    try:
        image_path = Path(image_path)

        # If path is not absolute, try to resolve it relative to PROJECT_ROOT
        if not image_path.is_absolute():
            image_path = Path(config.PROJECT_ROOT) / image_path

        if not image_path.exists():
            logger.error(f"Image not found: {image_path}")
            return None
        
        # Handle HEIC format
        if image_path.suffix.lower() == '.heic':
            img = Image.open(image_path)
            img = img.convert('RGB')
            return np.array(img)
        
        # Handle GIF (take first frame)
        elif image_path.suffix.lower() == '.gif':
            img = Image.open(image_path)
            img = img.convert('RGB')
            return np.array(img)
        
        # Handle other formats
        else:
            img = cv2.imread(str(image_path))
            if img is None:
                # Fallback to PIL
                img = Image.open(image_path)
                img = img.convert('RGB')
                return np.array(img)
            # Convert BGR to RGB
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
    except Exception as e:
        logger.error(f"Error loading image {image_path}: {str(e)}")
        return None


def get_image_files(directory: str) -> List[Path]:
    """
    Get all supported image files from a directory recursively.
    
    Args:
        directory: Path to the directory
        
    Returns:
        List of Path objects for image files
    """
    directory = Path(directory)
    image_files = []
    
    for ext in config.SUPPORTED_FORMATS:
        image_files.extend(directory.rglob(f"*{ext}"))
        image_files.extend(directory.rglob(f"*{ext.upper()}"))
    
    logger.info(f"Found {len(image_files)} image files in {directory}")
    return sorted(image_files)


def resize_image(image: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
    """
    Resize an image to target size while maintaining aspect ratio.
    
    Args:
        image: Input image as numpy array
        target_size: Tuple of (width, height)
        
    Returns:
        Resized image
    """
    h, w = image.shape[:2]
    target_w, target_h = target_size
    
    # Calculate scaling factor
    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)
    
    # Resize image
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    # Create canvas and center image
    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    y_offset = (target_h - new_h) // 2
    x_offset = (target_w - new_w) // 2
    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
    
    return canvas


def draw_face_box(image: np.ndarray, box: List[int], label: str = "") -> np.ndarray:
    """
    Draw a bounding box and label on an image.
    
    Args:
        image: Input image
        box: Bounding box coordinates [x, y, w, h]
        label: Text label to display
        
    Returns:
        Image with drawn box
    """
    img_copy = image.copy()
    x, y, w, h = box
    
    # Draw rectangle
    cv2.rectangle(img_copy, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    # Draw label
    if label:
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.rectangle(img_copy, (x, y - label_size[1] - 10), 
                     (x + label_size[0], y), (0, 255, 0), -1)
        cv2.putText(img_copy, label, (x, y - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    
    return img_copy


def save_face_crop(image: np.ndarray, box: List[int], output_path: str) -> bool:
    """
    Extract and save a face crop from an image.
    
    Args:
        image: Input image
        box: Bounding box coordinates [x, y, w, h]
        output_path: Path to save the cropped face
        
    Returns:
        True if successful, False otherwise
    """
    try:
        x, y, w, h = box
        
        # Add padding
        padding = 10
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(image.shape[1] - x, w + 2 * padding)
        h = min(image.shape[0] - y, h + 2 * padding)
        
        # Extract face
        face = image[y:y+h, x:x+w]
        
        # Save as JPEG
        face_rgb = cv2.cvtColor(face, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(output_path), face_rgb)
        
        return True
    except Exception as e:
        logger.error(f"Error saving face crop: {str(e)}")
        return False


def calculate_iou(box1: List[int], box2: List[int]) -> float:
    """
    Calculate Intersection over Union (IoU) between two bounding boxes.
    
    Args:
        box1: First bounding box [x, y, w, h]
        box2: Second bounding box [x, y, w, h]
        
    Returns:
        IoU score between 0 and 1
    """
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    
    # Calculate intersection
    x_left = max(x1, x2)
    y_top = max(y1, y2)
    x_right = min(x1 + w1, x2 + w2)
    y_bottom = min(y1 + h1, y2 + h2)
    
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    
    # Calculate union
    box1_area = w1 * h1
    box2_area = w2 * h2
    union_area = box1_area + box2_area - intersection_area
    
    return intersection_area / union_area if union_area > 0 else 0.0