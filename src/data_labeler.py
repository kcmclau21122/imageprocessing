"""
Interactive data labeling tool for tagging faces with names
Allows users to review detected faces and assign names
Updated with better error handling and console-first approach
"""
import json
import logging
from pathlib import Path
from typing import Dict, List
import numpy as np
from tqdm import tqdm

from . import config
from .face_detection import FaceDetector
from .utils import get_image_files, load_image

logger = logging.getLogger(__name__)

# Check if OpenCV GUI is available
try:
    import cv2
    # Test if GUI functions are available
    test_window = "test_opencv_gui"
    cv2.namedWindow(test_window, cv2.WINDOW_NORMAL)
    cv2.destroyWindow(test_window)
    OPENCV_GUI_AVAILABLE = True
except Exception:
    OPENCV_GUI_AVAILABLE = False
    cv2 = None
    logger.warning("OpenCV GUI not available - using console interface only")


class DataLabeler:
    """
    Interactive tool for labeling detected faces with names.
    Defaults to console interface (GUI optional if OpenCV supports it).
    """
    
    def __init__(self, training_dir: str = None, max_image_size: int = None):
        """
        Initialize data labeler.
        
        Args:
            training_dir: Directory containing training images
            max_image_size: Maximum image dimension for processing
        """
        self.training_dir = Path(training_dir or config.TRAINING_DIR)
        max_size = max_image_size or getattr(config, 'MAX_IMAGE_SIZE', 1920)
        self.detector = FaceDetector(max_image_size=max_size)
        
        self.annotations = {}
        self.annotation_file = self.training_dir / "annotations.json"
        self.skipped_images = []
        
        # Load existing annotations
        self.load_annotations()
    
    def load_annotations(self):
        """Load existing annotations from file."""
        if self.annotation_file.exists():
            try:
                with open(self.annotation_file, 'r') as f:
                    self.annotations = json.load(f)
                logger.info(f"Loaded {len(self.annotations)} existing annotations")
                
                # Print summary
                labeled_faces = sum(
                    sum(1 for face in faces if face.get('label'))
                    for faces in self.annotations.values()
                )
                skipped_faces = sum(
                    sum(1 for face in faces if face.get('label') is None)
                    for faces in self.annotations.values()
                )
                print(f"\n✓ Resuming from previous session:")
                print(f"  • {len(self.annotations)} images already processed")
                print(f"  • {labeled_faces} faces labeled")
                print(f"  • {skipped_faces} faces skipped (marked as unknown)")
                print(f"  • All previously processed images will be skipped\n")
                
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
        Label faces using console interface (recommended).
        This is the primary labeling method.
        """
        print("\n" + "="*70)
        print(" "*15 + "FACE LABELING TOOL - CONSOLE MODE")
        print("="*70)
        print("\nInstructions:")
        print("  • For each detected face, enter the person's name")
        print("  • Press ENTER to skip a face (if unsure)")
        print("  • Type 'unknown' for faces you can't identify")
        print("  • Type 'quit' to exit and save progress")
        print("  • Progress auto-saves after each image")
        print("="*70 + "\n")
        
        # Get all images
        image_files = get_image_files(self.training_dir)
        
        if not image_files:
            print(f"❌ No images found in {self.training_dir}")
            print(f"   Please add training images to this directory")
            return
        
        print(f"📸 Found {len(image_files)} total images\n")
        
        # Track statistics
        total_faces = 0
        labeled_faces = 0
        processed_images = 0
        
        for image_path in tqdm(image_files, desc="Processing images"):
            image_path_str = str(image_path)

            # Skip if already fully processed (all faces have been labeled or explicitly marked as unknown)
            if image_path_str in self.annotations:
                # Check if all faces have been processed (label set to something, even None/unknown)
                # An image is fully processed if we have annotations for it
                if len(self.annotations[image_path_str]) > 0:
                    # Skip this image - it's already been processed
                    continue

            # Detect faces with error handling (only once per image)
            try:
                faces = self.detector.detect_faces(image_path_str)
            except MemoryError:
                print(f"\n⚠️  Memory error: {image_path.name}")
                print(f"    Image too large - skipping...")
                self.skipped_images.append(str(image_path))
                continue
            except Exception as e:
                print(f"\n⚠️  Error processing {image_path.name}: {str(e)}")
                self.skipped_images.append(str(image_path))
                continue

            if not faces:
                # Mark image as processed with no faces
                self.annotations[image_path_str] = []
                continue

            processed_images += 1

            # Load image for info
            image = load_image(image_path_str)

            print(f"\n{'='*70}")
            print(f"📷 Image: {image_path.name}")
            if image is not None:
                print(f"   Size: {image.shape[1]}x{image.shape[0]}px")
            print(f"   Detected {len(faces)} face(s)")
            print(f"{'='*70}")

            # Initialize annotations for this image
            self.annotations[image_path_str] = []

            # Label each face
            for i, face in enumerate(faces):
                total_faces += 1
                
                x, y, w, h = face['box']
                confidence = face['confidence']
                
                print(f"\n👤 Face {i+1}/{len(faces)}:")
                print(f"   Location: ({x}, {y})")
                print(f"   Size: {w}x{h} pixels")
                print(f"   Confidence: {confidence:.1%}")
                
                # Get label from user
                while True:
                    label = input("   Enter name (ENTER=skip, 'quit'=exit): ").strip()

                    if label.lower() == 'quit':
                        print("\n💾 Saving and exiting...")
                        self.save_annotations()
                        self._print_summary(processed_images, total_faces, labeled_faces)
                        return

                    if label == '' or label.lower() == 'unknown':
                        label = None
                        print("   ⏭️  Skipped")
                        break

                    # Validate label
                    if len(label) < 2:
                        print("   ⚠️  Name too short - please enter at least 2 characters")
                        continue

                    if label in ['\\', '/', '.', '..']:
                        print("   ⚠️  Invalid name - please enter a valid person name")
                        continue

                    # Normalize label (capitalize first letter of each word)
                    label = label.title()

                    # Confirm label
                    confirm = input(f"   Confirm '{label}'? (y/n): ").strip().lower()
                    if confirm == 'y' or confirm == '':
                        labeled_faces += 1
                        print(f"   ✅ Labeled as: {label}")
                        break
                    else:
                        print("   Let's try again...")
                
                # Store annotation
                self.annotations[image_path_str].append({
                    'face_id': i,
                    'box': face['box'],
                    'confidence': confidence,
                    'label': label
                })
            
            # Auto-save after each image
            self.save_annotations()
            print(f"\n💾 Progress saved!")
        
        self._print_summary(processed_images, total_faces, labeled_faces)
    
    def label_images_visual(self):
        """
        Label faces using visual interface (requires OpenCV GUI support).
        Falls back to console if GUI not available.
        """
        if not OPENCV_GUI_AVAILABLE or cv2 is None:
            print("\n⚠️  OpenCV GUI not available!")
            print("   Reason: opencv-python was built without GUI support")
            print("   Solution: Using console interface instead\n")
            print("   To fix OpenCV GUI for future use:")
            print("   1. Uninstall: pip uninstall opencv-python")
            print("   2. Install: pip install opencv-contrib-python")
            print()
            self.label_images_console()
            return
        
        print("\n" + "="*70)
        print(" "*15 + "FACE LABELING TOOL - VISUAL MODE")
        print("="*70)
        print("\nInstructions:")
        print("  • A window will show each detected face")
        print("  • Enter the person's name in the console")
        print("  • Press any key in the image window to continue")
        print("  • Type 'quit' in console to exit")
        print("="*70 + "\n")
        
        # Get all images
        image_files = get_image_files(self.training_dir)
        
        if not image_files:
            print(f"❌ No images found in {self.training_dir}")
            return
        
        total_faces = 0
        labeled_faces = 0
        processed_images = 0
        
        try:
            for image_path in image_files:
                image_path_str = str(image_path)

                # Skip if already fully processed
                if image_path_str in self.annotations:
                    if len(self.annotations[image_path_str]) > 0:
                        continue

                # Detect faces
                try:
                    faces = self.detector.detect_faces(image_path_str)
                except Exception as e:
                    print(f"\n⚠️  Error: {image_path.name}: {str(e)}")
                    self.skipped_images.append(str(image_path))
                    continue

                if not faces:
                    # Mark as processed with no faces
                    self.annotations[image_path_str] = []
                    continue

                processed_images += 1
                print(f"\n📷 {image_path.name} - {len(faces)} face(s)")

                # Initialize annotations
                self.annotations[image_path_str] = []
                
                # Label each face
                for i, face in enumerate(faces):
                    total_faces += 1
                    face_img = face['face_img']
                    
                    # Resize for display
                    display_size = 400
                    h, w = face_img.shape[:2]
                    scale = display_size / max(h, w)
                    new_w, new_h = int(w * scale), int(h * scale)
                    face_display = cv2.resize(face_img, (new_w, new_h))
                    face_bgr = cv2.cvtColor(face_display, cv2.COLOR_RGB2BGR)
                    
                    # Show face
                    window_name = f'Face {i+1}/{len(faces)} - Press any key'
                    cv2.imshow(window_name, face_bgr)
                    cv2.waitKey(100)
                    
                    # Get label
                    label = input(f"👤 Face {i+1}/{len(faces)} - Name (ENTER=skip, 'quit'=exit): ").strip()
                    
                    if label.lower() == 'quit':
                        cv2.destroyAllWindows()
                        self.save_annotations()
                        self._print_summary(processed_images, total_faces, labeled_faces)
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
            
            self._print_summary(processed_images, total_faces, labeled_faces)
            
        except Exception as e:
            print(f"\n❌ Visual labeling error: {str(e)}")
            print("   Falling back to console interface...\n")
            self.label_images_console()
        finally:
            try:
                cv2.destroyAllWindows()
            except:
                pass
    
    def _print_summary(self, processed: int, total: int, labeled: int):
        """Print session summary."""
        print(f"\n{'='*70}")
        print(f"{'🎉 LABELING SESSION COMPLETE!':^70}")
        print(f"{'='*70}")
        print(f"\n📊 Statistics:")
        print(f"   • Images processed this session: {processed}")
        print(f"   • Total faces detected: {total}")
        print(f"   • Faces labeled: {labeled}")
        
        if labeled > 0:
            print(f"   • Labeling rate: {labeled/total:.1%}")
        
        if self.skipped_images:
            print(f"\n⚠️  Skipped {len(self.skipped_images)} images due to errors")
            if len(self.skipped_images) <= 5:
                for img in self.skipped_images:
                    print(f"      • {Path(img).name}")
            else:
                for img in self.skipped_images[:3]:
                    print(f"      • {Path(img).name}")
                print(f"      ... and {len(self.skipped_images) - 3} more")
            print(f"\n💡 Tip: Reduce MAX_IMAGE_SIZE in config.py if memory issues persist")
        
        print(f"\n💾 Annotations saved to: {self.annotation_file}")
        
        # Show what's labeled
        people = {}
        for faces in self.annotations.values():
            for face in faces:
                if face.get('label'):
                    people[face['label']] = people.get(face['label'], 0) + 1
        
        if people:
            print(f"\n👥 People labeled so far ({len(people)} unique):")
            for person, count in sorted(people.items(), key=lambda x: x[1], reverse=True):
                print(f"   • {person}: {count} faces")
        
        print(f"\n{'='*70}\n")
    
    def get_labeled_data(self) -> Dict[str, List]:
        """Get all labeled data organized by person."""
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
        """Print overall summary."""
        labeled_data = self.get_labeled_data()
        
        print(f"\n{'='*70}")
        print(f"{'📊 OVERALL LABELING SUMMARY':^70}")
        print(f"{'='*70}")
        print(f"\n📁 Total images processed: {len(self.annotations)}")
        print(f"👥 Unique people identified: {len(labeled_data)}")
        
        if labeled_data:
            print(f"\n📋 Faces per person:")
            for person, faces in sorted(labeled_data.items(), 
                                       key=lambda x: len(x[1]), 
                                       reverse=True):
                count = len(faces)
                status = "✅" if count >= 5 else "⚠️"
                print(f"   {status} {person}: {count} faces")
            
            # Recommendations
            needs_more = [p for p, f in labeled_data.items() if len(f) < 5]
            if needs_more:
                print(f"\n💡 Recommendation:")
                print(f"   Add more photos of: {', '.join(needs_more)}")
                print(f"   Target: 5-10 faces per person for best accuracy")
        
        if self.skipped_images:
            print(f"\n⚠️  {len(self.skipped_images)} images skipped")
        
        print(f"\n{'='*70}\n")


def main():
    """Main entry point."""
    print("\n" + "="*70)
    print(f"{'🏷️  FAMILY PHOTO FACE LABELING TOOL':^70}")
    print("="*70)
    
    # Get directory
    print("\n📂 Select training directory:")
    custom_dir = input("   Enter folder path (or ENTER for default): ").strip()
    
    if not custom_dir:
        custom_dir = str(config.TRAINING_DIR)
        print(f"   Using default: {custom_dir}")
    
    custom_path = Path(custom_dir)
    if not custom_path.exists():
        print(f"\n❌ Directory not found: {custom_dir}")
        print(f"   Please create the directory and add images first")
        return
    
    # Initialize labeler
    labeler = DataLabeler(training_dir=custom_path)
    
    # Choose interface
    print(f"\n🖥️  Select labeling interface:")
    print("   1. Console interface (recommended, always works)")
    print("   2. Visual interface (shows face images)")
    
    choice = input("\n   Enter choice (1 or 2, default=1): ").strip()
    
    print()  # Blank line
    
    if choice == '2':
        labeler.label_images_visual()
    else:
        labeler.label_images_console()
    
    # Print final summary
    labeler.print_summary()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user. Progress has been saved.")
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()