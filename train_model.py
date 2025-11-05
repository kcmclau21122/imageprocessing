"""
Training script for face recognition model
Loads labeled data and trains the classifier
"""
import json
import logging
from pathlib import Path
from typing import Dict, List
import numpy as np
from tqdm import tqdm

import config
from src.face_recognition import FaceRecognizer
from src.utils import load_image

logger = logging.getLogger(__name__)


class ModelTrainer:
    """
    Trainer for face recognition model.
    """
    
    def __init__(self, training_dir: str = None, model_name: str = None):
        """
        Initialize trainer.
        
        Args:
            training_dir: Directory containing training images and annotations
            model_name: Embedding model name
        """
        self.training_dir = Path(training_dir or config.TRAINING_DIR)
        self.annotation_file = self.training_dir / "annotations.json"
        self.recognizer = FaceRecognizer(model_name)
        self.annotations = {}
    
    def load_annotations(self) -> bool:
        """
        Load annotations from file.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            if not self.annotation_file.exists():
                logger.error(f"Annotations file not found: {self.annotation_file}")
                return False
            
            with open(self.annotation_file, 'r') as f:
                self.annotations = json.load(f)
            
            logger.info(f"Loaded annotations from {self.annotation_file}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading annotations: {str(e)}")
            return False
    
    def prepare_training_data(self) -> Dict[str, int]:
        """
        Extract faces and embeddings from labeled data.
        
        Returns:
            Dictionary with statistics about the training data
        """
        stats = {
            'total_images': 0,
            'total_faces': 0,
            'labeled_faces': 0,
            'unique_people': 0,
            'failed_extractions': 0
        }
        
        print("\nPreparing training data...")
        
        labeled_data = {}
        
        # Collect all labeled faces
        for image_path, faces in self.annotations.items():
            stats['total_images'] += 1
            
            for face in faces:
                stats['total_faces'] += 1
                
                label = face.get('label')
                if not label:
                    continue
                
                stats['labeled_faces'] += 1
                
                if label not in labeled_data:
                    labeled_data[label] = []
                
                labeled_data[label].append({
                    'image_path': image_path,
                    'box': face['box']
                })
        
        stats['unique_people'] = len(labeled_data)
        
        print(f"Found {stats['labeled_faces']} labeled faces for {stats['unique_people']} people")
        
        # Extract embeddings for each face
        print("\nExtracting face embeddings...")
        
        for person, face_list in tqdm(labeled_data.items(), desc="Processing people"):
            for face_info in tqdm(face_list, desc=f"  {person}", leave=False):
                # Load image
                image = load_image(face_info['image_path'])
                
                if image is None:
                    stats['failed_extractions'] += 1
                    continue
                
                # Extract face region
                x, y, w, h = face_info['box']
                face_img = image[y:y+h, x:x+w]
                
                # Add to training data
                success = self.recognizer.add_face(face_img, person)
                
                if not success:
                    stats['failed_extractions'] += 1
        
        return stats
    
    def train(self, classifier_type: str = 'svm', test_split: float = 0.2) -> Dict:
        """
        Train the face recognition model.
        
        Args:
            classifier_type: Type of classifier to use
            test_split: Fraction of data to use for testing
            
        Returns:
            Dictionary with training results
        """
        print("\n" + "="*60)
        print("TRAINING FACE RECOGNITION MODEL")
        print("="*60)
        
        # Load annotations
        if not self.load_annotations():
            return {'success': False, 'error': 'Failed to load annotations'}
        
        # Prepare training data
        stats = self.prepare_training_data()
        
        print(f"\nTraining data statistics:")
        print(f"  Total images: {stats['total_images']}")
        print(f"  Total faces detected: {stats['total_faces']}")
        print(f"  Labeled faces: {stats['labeled_faces']}")
        print(f"  Unique people: {stats['unique_people']}")
        print(f"  Failed extractions: {stats['failed_extractions']}")
        
        if stats['labeled_faces'] == 0:
            print("\nNo labeled data found. Please run the labeling tool first.")
            return {'success': False, 'error': 'No labeled data'}
        
        # Split data for validation
        if test_split > 0:
            print(f"\nSplitting data ({int((1-test_split)*100)}% train, {int(test_split*100)}% test)...")
            train_embeddings, train_labels, test_embeddings, test_labels = self._split_data(test_split)
        else:
            train_embeddings = self.recognizer.embeddings
            train_labels = self.recognizer.labels
            test_embeddings = []
            test_labels = []
        
        print(f"Training samples: {len(train_embeddings)}")
        print(f"Testing samples: {len(test_embeddings)}")
        
        # Train classifier
        print(f"\nTraining {classifier_type} classifier...")
        
        # Temporarily set embeddings to training set
        original_embeddings = self.recognizer.embeddings
        original_labels = self.recognizer.labels
        self.recognizer.embeddings = train_embeddings
        self.recognizer.labels = train_labels
        
        success = self.recognizer.train(classifier_type)
        
        # Restore full dataset
        self.recognizer.embeddings = original_embeddings
        self.recognizer.labels = original_labels
        
        if not success:
            return {'success': False, 'error': 'Training failed'}
        
        # Evaluate on test set if available
        results = {'success': True, 'stats': stats}
        
        if test_embeddings:
            print("\nEvaluating on test set...")
            eval_results = self.recognizer.evaluate(test_embeddings, test_labels)
            
            if eval_results:
                accuracy = eval_results['accuracy']
                print(f"\nTest Accuracy: {accuracy:.2%}")
                
                results['test_accuracy'] = accuracy
                results['evaluation'] = eval_results
                
                # Check if meets requirement
                if accuracy >= 0.75:
                    print(f"✓ Model meets the 75% accuracy requirement!")
                else:
                    print(f"✗ Model accuracy below 75% threshold")
        
        # Save model
        print("\nSaving model...")
        self.recognizer.save_model()
        print(f"Model saved to {config.MODELS_DIR}")
        
        print("\n" + "="*60)
        print("TRAINING COMPLETE")
        print("="*60 + "\n")
        
        return results
    
    def _split_data(self, test_split: float) -> tuple:
        """
        Split data into training and testing sets.
        
        Args:
            test_split: Fraction of data to use for testing
            
        Returns:
            Tuple of (train_embeddings, train_labels, test_embeddings, test_labels)
        """
        from sklearn.model_selection import train_test_split
        
        embeddings = np.array(self.recognizer.embeddings)
        labels = np.array(self.recognizer.labels)
        
        # Stratified split to maintain class distribution
        train_emb, test_emb, train_lbl, test_lbl = train_test_split(
            embeddings, labels,
            test_size=test_split,
            stratify=labels,
            random_state=42
        )
        
        return (
            train_emb.tolist(),
            train_lbl.tolist(),
            test_emb.tolist(),
            test_lbl.tolist()
        )
    
    def cross_validate(self, n_splits: int = 5) -> Dict:
        """
        Perform cross-validation on the model.
        
        Args:
            n_splits: Number of CV folds
            
        Returns:
            Dictionary with CV results
        """
        from sklearn.model_selection import cross_val_score
        
        print(f"\nPerforming {n_splits}-fold cross-validation...")
        
        if not self.load_annotations():
            return {'success': False}
        
        self.prepare_training_data()
        
        X = np.array(self.recognizer.embeddings)
        y = self.recognizer.label_encoder.fit_transform(self.recognizer.labels)
        
        # Initialize classifier
        from sklearn.svm import SVC
        classifier = SVC(kernel='linear', probability=True)
        
        # Perform CV
        scores = cross_val_score(classifier, X, y, cv=n_splits, scoring='accuracy')
        
        print(f"Cross-validation scores: {scores}")
        print(f"Mean accuracy: {scores.mean():.2%} (+/- {scores.std() * 2:.2%})")
        
        return {
            'success': True,
            'scores': scores.tolist(),
            'mean_accuracy': scores.mean(),
            'std_accuracy': scores.std()
        }


def main():
    """Main function for training the model."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train face recognition model')
    parser.add_argument('--model', type=str, default='Facenet512',
                       help='Embedding model (Facenet512, ArcFace, VGG-Face, etc.)')
    parser.add_argument('--classifier', type=str, default='svm',
                       choices=['svm', 'knn', 'random_forest'],
                       help='Classifier type')
    parser.add_argument('--test-split', type=float, default=0.2,
                       help='Fraction of data for testing (0.0 to 1.0)')
    parser.add_argument('--cross-validate', action='store_true',
                       help='Perform cross-validation')
    
    args = parser.parse_args()
    
    # Initialize trainer
    trainer = ModelTrainer(model_name=args.model)
    
    if args.cross_validate:
        # Run cross-validation
        results = trainer.cross_validate()
    else:
        # Train model
        results = trainer.train(
            classifier_type=args.classifier,
            test_split=args.test_split
        )
    
    return results


if __name__ == "__main__":
    main()
    