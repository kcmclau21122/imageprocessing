"""
Interactive data labeling tool for tagging faces with names
Allows users to review detected faces and assign names
"""
import json
import logging
from pathlib import Path
from typing import Dict, List
import cv2
import numpy as np
from tqdm import tqdm

import config
from src.face_detection import FaceDetector
from src.utils import get_image_files, load_image, draw_face_box

logger = logging.getLogger(__name__)


class DataLabeler:
    """
    Interactive tool for labeling detected faces with names.
    """
    
    def __init__(self, training_dir: str = None):
        """
        Initialize data labeler.
        
        Args:
            training_dir: Directory containing training images
        """
        self.training_dir = Path(training_dir or config.TRAINING_DIR)
        self.detector = FaceDetector()
        self.annotations = {}
        self.annotation_file = self.training_dir / "annotations.json"
        
        # Load existing annotations if available
        self.load_annotations()
    
    def load_annotations(self):
        """Load existing annotations from file."""
        if self.annotation_file.exists():
            try:
                with open(self.annotation_file, 'r') as f:
                    self.annotations = json.load(f)
                logger.info(f"Loaded {len(self.annotations)} existing annotations")
            except Exception as e:
                logger.error(f"Error loading annotations: {str(e)}")
                self.annotations = {}
    
    def save_annotations(self):
        """Save annotations to file."""
        try:
            with open(self.annotation_file, 'w') as f:
                json.dump(self.annotations, f, indent=2)
            logger.info(f"Saved annotations to {self.annotation_file}")
        except Exception as e:
            logger.error(f"Error saving annotations: {str(e)}")
    
    def label_images_console(self):
        """
        Label faces in images using console interface.
        This is the main method for interactive labeling.
        """
        print("\n" + "="*60)
        print("FAMILY PHOTO FACE LABELING TOOL")
        print("="*60)
        print("Instructions:")
        print("  - For each detected face, enter the person's name")
        print("  - Press ENTER to skip a face")
        print("  - Type 'unknown' for faces you can't identify")
        print("  - Type 'quit' to exit")
        print("="*60 + "\n")
        
        # Get all images
        image_files = get_image_files(self.training_dir)
        
        if not image_files:
            print(f"No images found in {self.training_dir}")
            return
        
        print(f"Found {len(image_files)} images\n")
        
        # Process each image
        total_faces = 0
        labeled_faces = 0
        
        for image_path in tqdm(image_files, desc="Processing images"):
            image_path_str = str(image_path)
            
            # Skip if already processed
            if image_path_str in self.annotations:
                existing_labels = [f for f in self.annotations[image_path_str] if f.get('label')]
                if existing_labels:
                    continue
            
            # Detect faces
            faces = self.detector.detect_faces(image_path_str)
            
            if not faces:
                continue
            
            # Load image for display info
            image = load_image(image_path_str)
            
            print(f"\n{'='*60}")
            print(f"Image: {image_path.name}")
            print(f"Detected {len(faces)} face(s)")
            print(f"{'='*60}")
            
            # Initialize annotations for this image
            if image_path_str not in self.annotations:
                self.annotations[image_path_str] = []
            
            # Label each face
            for i, face in enumerate(faces):
                total_faces += 1
                
                x, y, w, h = face['box']
                confidence = face['confidence']
                
                print(f"\nFace {i+1}/{len(faces)}:")
                print(f"  Location: ({x}, {y}) Size: {w}x{h}")
                print(f"  Confidence: {confidence:.2%}")
                
                # Get label from user
                while True:
                    label = input("  Enter name (or ENTER to skip, 'unknown', 'quit'): ").strip()
                    
                    if label.lower() == 'quit':
                        print("\nSaving and exiting...")
                        self.save_annotations()
                        return
                    
                    if label == '' or label.lower() == 'unknown':
                        label = None
                        break
                    
                    # Confirm label
                    confirm = input(f"  Confirm '{label}'? (y/n): ").strip().lower()
                    if confirm == 'y':
                        labeled_faces += 1
                        break
                
                # Store annotation
                self.annotations[image_path_str].append({
                    'face_id': i,
                    'box': face['box'],
                    'confidence': confidence,
                    'label': label
                })
            
            # Auto-save after each image
            self.save_annotations()
        
        print(f"\n{'='*60}")
        print(f"Labeling complete!")
        print(f"Total faces detected: {total_faces}")
        print(f"Faces labeled: {labeled_faces}")
        print(f"Annotations saved to: {self.annotation_file}")
        print(f"{'='*60}\n")
    
    def label_images_visual(self):
        """
        Label faces using OpenCV visual interface.
        Shows each face in a window for labeling.
        """
        print("\n" + "="*60)
        print("VISUAL FACE LABELING TOOL")
        print("="*60)
        print("Instructions:")
        print("  - A window will show each detected face")
        print("  - Enter the person's name in the console")
        print("  - Press any key in the image window to continue")
        print("  - Type 'quit' in console to exit")
        print("="*60 + "\n")
        
        # Get all images
        image_files = get_image_files(self.training_dir)
        
        if not image_files:
            print(f"No images found in {self.training_dir}")
            return
        
        total_faces = 0
        labeled_faces = 0
        
        for image_path in image_files:
            image_path_str = str(image_path)
            
            # Skip if already processed
            if image_path_str in self.annotations:
                existing_labels = [f for f in self.annotations[image_path_str] if f.get('label')]
                if existing_labels:
                    continue
            
            # Detect faces
            faces = self.detector.detect_faces(image_path_str)
            
            if not faces:
                continue
            
            # Load image
            image = load_image(image_path_str)
            
            print(f"\nImage: {image_path.name} - {len(faces)} face(s)")
            
            # Initialize annotations
            if image_path_str not in self.annotations:
                self.annotations[image_path_str] = []
            
            # Label each face
            for i, face in enumerate(faces):
                total_faces += 1
                
                # Get face image
                face_img = face['face_img']
                
                # Resize for display
                display_size = 300
                h, w = face_img.shape[:2]
                scale = display_size / max(h, w)
                new_w, new_h = int(w * scale), int(h * scale)
                face_display = cv2.resize(face_img, (new_w, new_h))
                
                # Convert to BGR for OpenCV
                face_display_bgr = cv2.cvtColor(face_display, cv2.COLOR_RGB2BGR)
                
                # Show face
                cv2.imshow(f'Face {i+1}/{len(faces)}', face_display_bgr)
                cv2.waitKey(100)  # Brief pause to display
                
                # Get label
                label = input(f"Face {i+1}/{len(faces)} - Enter name (ENTER=skip, 'quit'=exit): ").strip()
                
                if label.lower() == 'quit':
                    cv2.destroyAllWindows()
                    self.save_annotations()
                    return
                
                if label and label.lower() != 'unknown':
                    labeled_faces += 1
                else:
                    label = None
                
                # Store annotation
                self.annotations[image_path_str].append({
                    'face_id': i,
                    'box': face['box'],
                    'confidence': face['confidence'],
                    'label': label
                })
                
                cv2.destroyAllWindows()
            
            # Auto-save
            self.save_annotations()
        
        print(f"\nLabeling complete! Labeled {labeled_faces}/{total_faces} faces")
        cv2.destroyAllWindows()
    
    def get_labeled_data(self) -> Dict[str, List]:
        """
        Get all labeled face data organized by person.
        
        Returns:
            Dictionary mapping person names to lists of (image_path, box) tuples
        """
        labeled_data = {}
        
        for image_path, faces in self.annotations.items():
            for face in faces:
                label = face.get('label')
                if label:
                    if label not in labeled_data:
                        labeled_data[label] = []
                    
                    labeled_data[label].append({
                        'image_path': image_path,
                        'box': face['box']
                    })
        
        return labeled_data
    
    def print_summary(self):
        """Print a summary of labeled data."""
        labeled_data = self.get_labeled_data()
        
        print("\n" + "="*60)
        print("LABELING SUMMARY")
        print("="*60)
        print(f"Total images processed: {len(self.annotations)}")
        print(f"Unique people identified: {len(labeled_data)}")
        print()
        
        for person, faces in sorted(labeled_data.items()):
            print(f"  {person}: {len(faces)} faces")
        
        print("="*60 + "\n")


def main():
    """Main function for running the labeling tool."""
    import sys
    
    # Check if training directory has images
    if not config.TRAINING_DIR.exists():
        print(f"Training directory not found: {config.TRAINING_DIR}")
        print("Please create the directory and add your training images.")
        return
    
    # Initialize labeler
    labeler = DataLabeler()
    
    # Choose interface
    print("\nSelect labeling interface:")
    print("1. Console interface (recommended)")
    print("2. Visual interface (shows faces in windows)")
    
    choice = input("Enter choice (1 or 2): ").strip()
    
    if choice == '2':
        labeler.label_images_visual()
    else:
        labeler.label_images_console()
    
    # Print summary
    labeler.print_summary()


if __name__ == "__main__":
    main()