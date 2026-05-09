"""
Train face recognition model using DeepFace embeddings and machine learning classifiers.
Now includes:
 - Albumentations-based image augmentation during training
 - Optional dlib-based face alignment for consistent embeddings
 - Class weighting for imbalanced data
 - Enhanced face detection and alignment
"""

import json
import logging
from pathlib import Path
import argparse
from typing import Dict, List, Tuple
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import classification_report, accuracy_score
from sklearn.utils.class_weight import compute_class_weight

import cv2
from src import config
from src.face_recognition import FaceRecognizer
from src.utils import load_image

logger = logging.getLogger(__name__)

# Optional dependencies
try:
    import dlib
    HAS_DLIB = True
except Exception:
    dlib = None
    HAS_DLIB = False

try:
    from albumentations import Compose, HorizontalFlip, RandomBrightnessContrast, Rotate, GaussianBlur, HueSaturationValue, RandomGamma, OneOf
    HAS_AUG = True
except Exception:
    Compose = None
    HAS_AUG = False


class ModelTrainer:
    """
    Handles preparing data, extracting embeddings, and training the recognizer.
    """

    def __init__(self, model_name: str = None, classifier_type: str = 'svm'):
        self.annotation_file = Path(config.TRAINING_DIR) / "annotations.json"
        self.model_name = model_name or config.EMBEDDING_MODEL
        self.classifier_type = classifier_type
        self.recognizer = FaceRecognizer(model_name=self.model_name)
        self.shape_predictor = None

        # Try loading dlib predictor
        if HAS_DLIB:
            predictor_path = Path(getattr(config, 'DLIB_SHAPE_PREDICTOR', 'shape_predictor_68_face_landmarks.dat'))
            if predictor_path.exists():
                try:
                    self.shape_predictor = dlib.shape_predictor(str(predictor_path))
                    logger.info(f"Loaded dlib shape predictor: {predictor_path}")
                except Exception as e:
                    logger.warning(f"Failed to load dlib predictor: {e}")
            else:
                logger.warning("dlib shape predictor not found — using simple cropping.")
        else:
            logger.warning("dlib not available — skipping alignment.")

        # Prepare enhanced augmentation
        if HAS_AUG:
            self.augment = Compose([
                HorizontalFlip(p=0.5),
                RandomBrightnessContrast(p=0.4, brightness_limit=0.2, contrast_limit=0.2),
                Rotate(limit=15, p=0.3),
                GaussianBlur(blur_limit=3, p=0.1),
                RandomGamma(gamma_limit=(80, 120), p=0.2),
                OneOf([
                    RandomBrightnessContrast(),
                    HueSaturationValue(),
                ], p=0.3),
            ])
            logger.info("Enhanced Albumentations augmentation enabled.")
        else:
            self.augment = None
            logger.warning("Albumentations not installed — skipping augmentation.")

    # -----------------------------
    # Enhanced Face alignment and cropping
    # -----------------------------
    def align_or_crop(self, img: np.ndarray, box: List[int]) -> np.ndarray:
        """Enhanced alignment with better error handling and expanded bounding box."""
        x, y, w, h = box
        
        # Skip very small faces
        if w < 50 or h < 50:
            logger.debug(f"Skipping small face: {w}x{h}")
            return None
        
        # Expand bounding box slightly for better context
        expand = 0.1
        x_exp = max(0, int(x - w * expand))
        y_exp = max(0, int(y - h * expand))
        w_exp = min(img.shape[1] - x_exp, int(w * (1 + 2 * expand)))
        h_exp = min(img.shape[0] - y_exp, int(h * (1 + 2 * expand)))
        
        ih, iw = img.shape[:2]
        x1, y1 = max(0, x_exp), max(0, y_exp)
        x2, y2 = min(iw, x_exp + w_exp), min(ih, y_exp + h_exp)

        if self.shape_predictor is not None:
            try:
                rect = dlib.rectangle(x1, y1, x2, y2)
                shape = self.shape_predictor(img, rect)
                chips = dlib.get_face_chips(img, [shape], size=160)
                if chips:
                    return chips[0]
            except Exception as e:
                logger.debug(f"Alignment failed for box {box}: {e}")

        # Fallback to expanded crop
        cropped = img[y1:y2, x1:x2]
        if cropped.size == 0:
            return None
        return cropped

    # -----------------------------
    # Enhanced Training data preparation
    # -----------------------------
    def prepare_training_data(self) -> Dict[str, int]:
        """Extract faces, apply augmentation, and compute embeddings."""
        stats = {
            'total_images': 0,
            'total_faces': 0,
            'labeled_faces': 0,
            'unique_people': 0,
            'failed_extractions': 0,
            'skipped_small_faces': 0,
            'skipped_invalid_labels': 0
        }

        try:
            with open(self.annotation_file, 'r') as f:
                annotations = json.load(f)
        except Exception:
            logger.error(f"Could not read {self.annotation_file}")
            return stats

        people = set()

        for image_path_str, faces in tqdm(annotations.items(), desc="Preparing faces"):
            stats['total_images'] += 1
            image = load_image(image_path_str)
            if image is None:
                stats['failed_extractions'] += len(faces)
                continue

            for face_info in faces:
                stats['total_faces'] += 1

                # Handle None labels and ensure it's a string
                person = face_info.get('label')

                # Skip None, null, or intentionally unlabeled faces
                if person is None or person == 'None' or person == 'null':
                    stats['skipped_invalid_labels'] += 1
                    continue

                # Convert to string and strip whitespace
                person = str(person).strip()

                # Skip empty strings or unknown variations (comprehensive validation)
                if not person or person.lower() in ['unknown', 'none', 'null', '']:
                    stats['skipped_invalid_labels'] += 1
                    continue

                # Skip invalid special characters
                if person in ['\\', '/', '.', '..']:
                    stats['skipped_invalid_labels'] += 1
                    continue

                # Skip labels that are too short (likely errors)
                if len(person) < 2:
                    stats['skipped_invalid_labels'] += 1
                    continue

                # Normalize the label (capitalize first letter of each word)
                person = person.title()

                people.add(person)
                stats['labeled_faces'] += 1

                try:
                    x, y, w, h = face_info['box']
                    face_img = self.align_or_crop(image, (x, y, w, h))
                    
                    if face_img is None:
                        stats['skipped_small_faces'] += 1
                        continue
                        
                    # Resize to consistent size if needed
                    if face_img.shape[0] < 50 or face_img.shape[1] < 50:
                        face_img = cv2.resize(face_img, (160, 160))
                        
                except Exception as e:
                    logger.debug(f"Face crop failed for {image_path_str}: {e}")
                    stats['failed_extractions'] += 1
                    continue

                # Add original face
                success = self.recognizer.add_face(face_img, person)
                if not success:
                    stats['failed_extractions'] += 1
                else:
                    # Add augmented versions (training only)
                    if self.augment is not None:
                        aug_count = 0
                        for _ in range(3):  # Increased from 2 to 3
                            try:
                                aug_img = self.augment(image=face_img)['image']
                                if self.recognizer.add_face(aug_img, person):
                                    aug_count += 1
                            except Exception as e:
                                logger.debug(f"Augmentation failed: {e}")
                        
                        if aug_count > 0:
                            logger.debug(f"Added {aug_count} augmented versions for {person}")

        stats['unique_people'] = len(people)
        
        # Log class distribution
        if self.recognizer.labels:
            from collections import Counter
            label_counts = Counter(self.recognizer.labels)
            logger.info("Class distribution:")
            for person, count in label_counts.most_common():
                logger.info(f"  {person}: {count} samples")
                
        return stats


    # -----------------------------
    # Enhanced Model training with class weighting
    # -----------------------------
    def train(self) -> Tuple[Dict, float]:
        """Train and evaluate model with class weighting."""
        stats = self.prepare_training_data()
        if stats['labeled_faces'] == 0:
            logger.error("No labeled faces found.")
            return stats, 0.0

        logger.info(f"Training {self.classifier_type} classifier on {stats['labeled_faces']} faces.")
        
        # Calculate class weights for imbalanced data
        class_weight = None
        if len(self.recognizer.labels) > 0:
            try:
                y_encoded = self.recognizer.label_encoder.fit_transform(self.recognizer.labels)
                class_weights = compute_class_weight(
                    'balanced', 
                    classes=np.unique(y_encoded), 
                    y=y_encoded
                )
                class_weight = dict(enumerate(class_weights))
                logger.info("Using class weights for imbalanced data")
            except Exception as e:
                logger.warning(f"Could not compute class weights: {e}")

        success = self.recognizer.train(
            classifier_type=self.classifier_type,
            class_weight=class_weight
        )
        
        accuracy = 0.0
        if success:
            logger.info("Model trained successfully.")
            accuracy = self.evaluate_split()
            
            # Also run cross-validation for better assessment
            if len(np.unique(self.recognizer.labels)) >= 5:  # Only if enough classes
                cv_accuracy = self.cross_validate()
                logger.info(f"Cross-validation accuracy: {cv_accuracy:.2%}")
        else:
            logger.error("Training failed.")
            
        return stats, accuracy

    # -----------------------------
    # Enhanced Cross-validation
    # -----------------------------
    def cross_validate(self, n_splits: int = 5) -> float:
        """Perform k-fold cross-validation with detailed metrics."""
        if len(self.recognizer.embeddings) == 0:
            return 0.0
            
        X = np.array(self.recognizer.embeddings)
        y = np.array(self.recognizer.labels)
        y_encoded = self.recognizer.label_encoder.fit_transform(y)

        if len(np.unique(y_encoded)) < 2:
            logger.warning("Not enough classes for cross-validation.")
            return 0.0

        # Use fewer splits if not enough data
        n_splits = min(n_splits, len(np.unique(y_encoded)))
        if n_splits < 2:
            return 0.0

        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        scores = []

        for train_idx, test_idx in skf.split(X, y_encoded):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y_encoded[train_idx], y_encoded[test_idx]

            # Create a fresh classifier for each fold
            from sklearn.svm import SVC
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.neighbors import KNeighborsClassifier
            
            if self.classifier_type == 'svm':
                classifier = SVC(C=1.0, kernel='rbf', gamma='scale', probability=True, random_state=42)
            elif self.classifier_type == 'random_forest':
                classifier = RandomForestClassifier(n_estimators=200, max_depth=20, random_state=42)
            elif self.classifier_type == 'knn':
                classifier = KNeighborsClassifier(n_neighbors=5, weights='distance')
            else:
                classifier = SVC(probability=True, random_state=42)
                
            classifier.fit(X_train, y_train)
            preds = classifier.predict(X_test)
            acc = accuracy_score(y_test, preds)
            scores.append(acc)

        mean_acc = float(np.mean(scores))
        std_acc = float(np.std(scores))
        logger.info(f"{n_splits}-fold cross-validation accuracy: {mean_acc:.2%} ± {std_acc:.2%}")
        return mean_acc

    # -----------------------------
    # Enhanced Holdout evaluation
    # -----------------------------
    def evaluate_split(self, test_size: float = 0.2) -> float:
        """Split embeddings and evaluate accuracy on holdout set."""
        if len(self.recognizer.embeddings) == 0:
            return 0.0
            
        X = np.array(self.recognizer.embeddings)
        y = np.array(self.recognizer.labels)
        y_encoded = self.recognizer.label_encoder.fit_transform(y)

        if len(np.unique(y_encoded)) < 2:
            logger.warning("Not enough classes for evaluation.")
            return 0.0

        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=test_size, stratify=y_encoded, random_state=42
        )

        self.recognizer.classifier.fit(X_train, y_train)
        preds = self.recognizer.classifier.predict(X_test)
        acc = accuracy_score(y_test, preds)
        
        # Get the actual classes present in the test data
        unique_labels = np.unique(np.concatenate([y_test, preds]))
        available_classes = self.recognizer.label_encoder.inverse_transform(unique_labels)
        
        report = classification_report(
            y_test, preds,
            target_names=available_classes,
            digits=3
        )

        print("\n" + "=" * 60)
        print("EVALUATION REPORT")
        print("=" * 60)
        print(report)
        print("=" * 60 + "\n")

        logger.info(f"Holdout accuracy: {acc:.2%}")
        return acc

    # -----------------------------
    # Ensemble Training (Optional)
    # -----------------------------
    def train_ensemble(self) -> bool:
        """Train an ensemble of classifiers for potentially better accuracy."""
        if len(self.recognizer.embeddings) == 0:
            return False
            
        X = np.array(self.recognizer.embeddings)
        y = np.array(self.recognizer.labels)
        y_encoded = self.recognizer.label_encoder.fit_transform(y)
        
        try:
            from sklearn.ensemble import VotingClassifier
            from sklearn.svm import SVC
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.neighbors import KNeighborsClassifier
            
            estimators = [
                ('svm', SVC(C=1.0, kernel='rbf', probability=True, random_state=42)),
                ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
                ('knn', KNeighborsClassifier(n_neighbors=5, weights='distance'))
            ]
            
            self.recognizer.classifier = VotingClassifier(estimators, voting='soft')
            self.recognizer.classifier.fit(X, y_encoded)
            
            logger.info("Ensemble model trained successfully")
            return True
            
        except Exception as e:
            logger.error(f"Ensemble training failed: {e}")
            # Fall back to single classifier
            return self.recognizer.train(classifier_type=self.classifier_type)


# -----------------------------
# CLI entry point
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description="Train face recognition model with augmentation/alignment")
    parser.add_argument("--model", type=str, default=None,
                        help="Embedding model (ArcFace, Facenet, etc.)")
    parser.add_argument("--classifier", type=str, default="svm",
                        choices=["svm", "knn", "random_forest"],
                        help="Classifier type")
    parser.add_argument("--cross-validate", action="store_true",
                        help="Run 5-fold cross-validation instead of holdout")
    parser.add_argument("--ensemble", action="store_true",
                        help="Use ensemble of classifiers (experimental)")
    args = parser.parse_args()

    trainer = ModelTrainer(model_name=args.model, classifier_type=args.classifier)
    
    if args.ensemble:
        stats = trainer.prepare_training_data()
        if stats['labeled_faces'] == 0:
            logger.error("No labeled faces found.")
            return
            
        success = trainer.train_ensemble()
        accuracy = trainer.evaluate_split() if success else 0.0
    else:
        stats, accuracy = trainer.train()

    if args.cross_validate:
        mean_acc = trainer.cross_validate()
        print(f"\nCross-validation mean accuracy: {mean_acc:.2%}")

    print("\n" + "=" * 60)
    print("TRAINING SUMMARY")
    print("=" * 60)
    for k, v in stats.items():
        print(f"{k:25}: {v}")
    
    # Calculate and display average samples per class
    if stats['unique_people'] > 0:
        avg_samples = stats['labeled_faces'] / stats['unique_people']
        print(f"{'Avg samples per class':25}: {avg_samples:.1f}")
    
    print("=" * 60 + "\n")

    # Save model if training was successful
    if accuracy > 0 or args.ensemble:
        trainer.recognizer.save_model()
        print("Model saved successfully.")
    else:
        print("Model not saved due to training issues.")


if __name__ == "__main__":
    main()
