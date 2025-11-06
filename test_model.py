"""
Testing and inference script for face identification
Identifies family members in new photos using trained model
"""
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple
import cv2
import numpy as np
from tqdm import tqdm

import config
from src.face_detection import FaceDetector
from src.face_recognition import FaceRecognizer
from src.utils import load_image, draw_face_box, get_image_files

logger = logging.getLogger(__name__)


class FaceIdentifier:
    """
    Identifier for recognizing faces in new photos.
    """
    
    def __init__(self):
        """
        Initialize face identifier.
        """
        self.detector = FaceDetector()
        # Initialize recognizer without specifying model name
        # This prevents it from trying to download ArcFace weights
        self.recognizer = FaceRecognizer(model_name=None)
        
        # Load trained model
        self.load_model()
    
    def load_model(self) -> bool:
        """
        Load the trained model.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Load the trained model (embeddings + classifier)
            self.recognizer.load_model()
            logger.info("Successfully loaded trained model")
            return True
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return False
    
    def identify_faces(self, image_path: str, 
                      confidence_threshold: float = None) -> List[Dict]:
        """
        Identify all faces in an image.
        
        Args:
            image_path: Path to the image
            confidence_threshold: Minimum confidence for identification
            
        Returns:
            List of dictionaries with face information and predictions
        """
        threshold = confidence_threshold or config.CONFIDENCE_THRESHOLD
        
        # Detect faces
        faces = self.detector.detect_faces(image_path)
        
        if not faces:
            logger.info(f"No faces detected in {image_path}")
            return []
        
        # Identify each face
        results = []
        
        for i, face in enumerate(faces):
            face_img = face['face_img']
            
            # Predict identity
            label, confidence = self.recognizer.predict(face_img)
            
            # Only include if meets confidence threshold
            identified = confidence >= threshold if label else False
            
            result = {
                'face_id': i,
                'box': face['box'],
                'detection_confidence': face['confidence'],
                'predicted_label': label if identified else 'Unknown',
                'prediction_confidence': confidence,
                'identified': identified
            }
            
            results.append(result)
        
        return results
    
    # ... (rest of your methods remain exactly the same)
    def identify_batch(self, image_paths: List[str], 
                      confidence_threshold: float = None) -> Dict[str, List[Dict]]:
        """
        Identify faces in multiple images.
        
        Args:
            image_paths: List of image paths
            confidence_threshold: Minimum confidence for identification
            
        Returns:
            Dictionary mapping image paths to face identification results
        """
        results = {}
        
        for image_path in tqdm(image_paths, desc="Identifying faces"):
            faces = self.identify_faces(image_path, confidence_threshold)
            if faces:
                results[str(image_path)] = faces
        
        return results
    
    def visualize_results(self, image_path: str, 
                         confidence_threshold: float = None,
                         save_path: str = None,
                         display: bool = True) -> np.ndarray:
        """
        Visualize identification results on the image.
        
        Args:
            image_path: Path to the image
            confidence_threshold: Minimum confidence for identification
            save_path: Optional path to save annotated image
            display: Whether to display the image
            
        Returns:
            Annotated image as numpy array
        """
        # Load image
        image = load_image(image_path)
        
        if image is None:
            logger.error(f"Failed to load image: {image_path}")
            return None
        
        # Identify faces
        results = self.identify_faces(image_path, confidence_threshold)
        
        # Draw boxes and labels
        for result in results:
            box = result['box']
            label = result['predicted_label']
            confidence = result['prediction_confidence']
            
            # Create label text
            if result['identified']:
                text = f"{label} ({confidence:.0%})"
                color = (0, 255, 0)  # Green for identified
            else:
                text = "Unknown"
                color = (255, 0, 0)  # Red for unknown
            
            # Draw bounding box
            x, y, w, h = box
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
            
            # Draw label background
            label_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(image, (x, y - label_size[1] - 10), 
                         (x + label_size[0], y), color, -1)
            
            # Draw label text
            cv2.putText(image, text, (x, y - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Save if requested
        if save_path:
            output_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(save_path, output_image)
            logger.info(f"Saved annotated image to {save_path}")
        
        # Display if requested
        if display:
            display_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.imshow('Face Identification', display_image)
            print("\nPress any key to close the image...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        return image
    
    def test_accuracy(self, test_dir: str = None) -> Dict:
        """
        Test model accuracy on labeled test data.
        
        Args:
            test_dir: Directory containing test images with annotations
            
        Returns:
            Dictionary with accuracy metrics
        """
        test_dir = Path(test_dir or config.TESTING_DIR)
        annotation_file = test_dir / "annotations.json"
        
        print("\n" + "="*60)
        print("TESTING MODEL ACCURACY")
        print("="*60)
        
        # Load test annotations
        if not annotation_file.exists():
            print(f"No annotations found in {test_dir}")
            print("Please label your test images first using the labeling tool.")
            return {'success': False}
        
        with open(annotation_file, 'r') as f:
            annotations = json.load(f)
        
        # Test each image
        total_faces = 0
        correct_predictions = 0
        per_person_stats = {}
        
        print(f"\nTesting on {len(annotations)} images...")
        
        for image_path, faces in tqdm(annotations.items()):
            # Identify faces
            predictions = self.identify_faces(image_path)
            
            # Match predictions with ground truth
            for face in faces:
                true_label = face.get('label')
                
                if not true_label:
                    continue
                
                total_faces += 1
                
                # Find matching prediction by IoU
                face_box = face['box']
                best_match = None
                best_iou = 0
                
                for pred in predictions:
                    from src.utils import calculate_iou
                    iou = calculate_iou(face_box, pred['box'])
                    
                    if iou > best_iou:
                        best_iou = iou
                        best_match = pred
                
                # Check if prediction is correct
                if best_match and best_iou > 0.5:
                    predicted_label = best_match['predicted_label']
                    
                    if predicted_label == true_label:
                        correct_predictions += 1
                        
                        # Update per-person stats
                        if true_label not in per_person_stats:
                            per_person_stats[true_label] = {'correct': 0, 'total': 0}
                        per_person_stats[true_label]['correct'] += 1
                    
                    # Update total for person
                    if true_label not in per_person_stats:
                        per_person_stats[true_label] = {'correct': 0, 'total': 0}
                    per_person_stats[true_label]['total'] += 1
        
        # Calculate metrics
        if total_faces == 0:
            print("No labeled test faces found")
            return {'success': False}
        
        overall_accuracy = correct_predictions / total_faces
        
        print("\n" + "="*60)
        print("TEST RESULTS")
        print("="*60)
        print(f"\nOverall Accuracy: {overall_accuracy:.2%}")
        print(f"Correct: {correct_predictions}/{total_faces}")
        
        print("\nPer-Person Accuracy:")
        for person, stats in sorted(per_person_stats.items()):
            person_accuracy = stats['correct'] / stats['total']
            print(f"  {person}: {person_accuracy:.2%} ({stats['correct']}/{stats['total']})")
        
        # Check if meets requirement
        if overall_accuracy >= 0.90:
            print(f"\n✓ Model EXCEEDS the 90% accuracy requirement!")
        elif overall_accuracy >= 0.75:
            print(f"\n✓ Model meets the 75% accuracy requirement")
            print(f"  (but below 90% target)")
        else:
            print(f"\n✗ Model accuracy below 75% threshold")
            print(f"  Consider adding more training data or adjusting parameters")
        
        print("="*60 + "\n")
        
        return {
            'success': True,
            'overall_accuracy': overall_accuracy,
            'correct_predictions': correct_predictions,
            'total_faces': total_faces,
            'per_person_stats': per_person_stats
        }
    
    def process_directory(self, input_dir: str, output_dir: str = None,
                         confidence_threshold: float = None):
        """
        Process all images in a directory and save annotated results.
        
        Args:
            input_dir: Directory containing images to process
            output_dir: Directory to save annotated images
            confidence_threshold: Minimum confidence for identification
        """
        input_dir = Path(input_dir)
        output_dir = Path(output_dir) if output_dir else input_dir / "identified"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\nProcessing images from: {input_dir}")
        print(f"Saving results to: {output_dir}")
        
        # Get image files
        image_files = get_image_files(input_dir)
        
        if not image_files:
            print("No images found")
            return
        
        # Process each image
        results_summary = []
        
        for image_path in tqdm(image_files, desc="Processing"):
            # Identify faces
            results = self.identify_faces(str(image_path), confidence_threshold)
            
            if results:
                # Save annotated image
                output_path = output_dir / f"{image_path.stem}_identified{image_path.suffix}"
                self.visualize_results(str(image_path), confidence_threshold,
                                     save_path=str(output_path), display=False)
                
                # Add to summary
                identified_people = [r['predicted_label'] for r in results if r['identified']]
                results_summary.append({
                    'image': image_path.name,
                    'total_faces': len(results),
                    'identified_faces': len(identified_people),
                    'people': identified_people
                })
        
        # Save summary
        summary_file = output_dir / "identification_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(results_summary, f, indent=2)
        
        print(f"\nProcessed {len(image_files)} images")
        print(f"Results saved to {output_dir}")
        print(f"Summary saved to {summary_file}")


def main():
    """Main function for testing/identification."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Identify faces in photos')
    parser.add_argument('--image', type=str, help='Single image to process')
    parser.add_argument('--directory', type=str, help='Directory of images to process')
    parser.add_argument('--test', action='store_true', help='Run accuracy test')
    parser.add_argument('--output', type=str, help='Output directory for annotated images')
    parser.add_argument('--confidence', type=float, default=0.75,
                       help='Confidence threshold (0.0 to 1.0)')
    parser.add_argument('--display', action='store_true', help='Display results')
    
    args = parser.parse_args()
    
    # Initialize identifier
    identifier = FaceIdentifier()
    
    # Check if model was loaded successfully
    if not identifier.recognizer.classifier:
        print("Error: No trained model found. Please train a model first using:")
        print("python train_model.py --model ArcFace --classifier svm")
        return
    
    if args.test:
        # Run accuracy test
        identifier.test_accuracy()
    
    elif args.image:
        # Process single image
        identifier.visualize_results(
            args.image,
            confidence_threshold=args.confidence,
            save_path=args.output,
            display=args.display or not args.output
        )
    
    elif args.directory:
        # Process directory
        identifier.process_directory(
            args.directory,
            output_dir=args.output,
            confidence_threshold=args.confidence
        )
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
