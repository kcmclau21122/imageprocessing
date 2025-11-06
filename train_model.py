"""
Train face recognition model using DeepFace embeddings and machine learning classifiers.
Now includes:
 - Albumentations-based image augmentation during training
 - Optional dlib-based face alignment for consistent embeddings
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

import cv2
import config
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
    from albumentations import Compose, HorizontalFlip, RandomBrightnessContrast, Rotate
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

        # Prepare augmentation
        if HAS_AUG:
            self.augment = Compose([
                HorizontalFlip(p=0.5),
                RandomBrightnessContrast(p=0.4),
                Rotate(limit=8, p=0.3)
            ])
            logger.info("Albumentations augmentation enabled.")
        else:
            self.augment = None
            logger.warning("Albumentations not installed — skipping augmentation.")

    # -----------------------------
    # Face alignment and cropping
    # -----------------------------
    def align_or_crop(self, img: np.ndarray, box: List[int]) -> np.ndarray:
        """Try to align using dlib, else fall back to simple crop."""
        x, y, w, h = box
        ih, iw = img.shape[:2]
        x1, y1 = max(0, int(x)), max(0, int(y))
        x2, y2 = min(iw, int(x + w)), min(ih, int(y + h))

        if self.shape_predictor is not None:
            try:
                rect = dlib.rectangle(x1, y1, x2, y2)
                shape = self.shape_predictor(img, rect)
                chips = dlib.get_face_chips(img, [shape], size=160)
                if chips:
                    return chips[0]
            except Exception as e:
                logger.debug(f"Alignment failed for box {box}: {e}")

        return img[y1:y2, x1:x2]

    # -----------------------------
    # Training data preparation
    # -----------------------------
    def prepare_training_data(self) -> Dict[str, int]:
        """Extract faces, apply augmentation, and compute embeddings."""
        stats = {
            'total_images': 0,
            'total_faces': 0,
            'labeled_faces': 0,
            'unique_people': 0,
            'failed_extractions': 0
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
                person = face_info.get('label')
                if not person:
                    continue
                people.add(person)
                stats['labeled_faces'] += 1

                try:
                    x, y, w, h = face_info['box']
                    face_img = self.align_or_crop(image, (x, y, w, h))
                except Exception as e:
                    logger.debug(f"Face crop failed for {image_path_str}: {e}")
                    stats['failed_extractions'] += 1
                    continue

                # Add original face
                success = self.recognizer.add_face(face_img, person)
                if not success:
                    stats['failed_extractions'] += 1

                # Add augmented versions (training only)
                if self.augment is not None:
                    for _ in range(2):
                        try:
                            aug_img = self.augment(image=face_img)['image']
                            self.recognizer.add_face(aug_img, person)
                        except Exception as e:
                            logger.debug(f"Augmentation failed: {e}")

        stats['unique_people'] = len(people)
        return stats

    # -----------------------------
    # Model training
    # -----------------------------
    def train(self) -> Tuple[Dict, float]:
        """Train and evaluate model."""
        stats = self.prepare_training_data()
        if stats['labeled_faces'] == 0:
            logger.error("No labeled faces found.")
            return stats, 0.0

        logger.info(f"Training {self.classifier_type} classifier on {stats['labeled_faces']} faces.")
        success = self.recognizer.train(classifier_type=self.classifier_type)
        accuracy = 0.0
        if success:
            logger.info("Model trained successfully.")
            accuracy = self.evaluate_split()
        else:
            logger.error("Training failed.")
        return stats, accuracy

    # -----------------------------
    # Cross-validation
    # -----------------------------
    def cross_validate(self, n_splits: int = 5) -> float:
        """Perform k-fold cross-validation."""
        X = np.array(self.recognizer.embeddings)
        y = np.array(self.recognizer.labels)
        y_encoded = self.recognizer.label_encoder.fit_transform(y)

        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        scores = []

        for train_idx, test_idx in skf.split(X, y_encoded):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y_encoded[train_idx], y_encoded[test_idx]

            self.recognizer.classifier.fit(X_train, y_train)
            preds = self.recognizer.classifier.predict(X_test)
            acc = accuracy_score(y_test, preds)
            scores.append(acc)

        mean_acc = float(np.mean(scores))
        logger.info(f"{n_splits}-fold cross-validation accuracy: {mean_acc:.2%}")
        return mean_acc

    # -----------------------------
    # Holdout evaluation
    # -----------------------------
    def evaluate_split(self, test_size: float = 0.2) -> float:
        """Split embeddings and evaluate accuracy on holdout set."""
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
            target_names=available_classes,  # Use only classes present in test data
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
    args = parser.parse_args()

    trainer = ModelTrainer(model_name=args.model, classifier_type=args.classifier)
    stats, _ = trainer.train()

    if args.cross_validate:
        mean_acc = trainer.cross_validate()
        print(f"\nCross-validation mean accuracy: {mean_acc:.2%}")

    print("\n" + "=" * 60)
    print("TRAINING SUMMARY")
    print("=" * 60)
    for k, v in stats.items():
        print(f"{k:20}: {v}")
    print("=" * 60 + "\n")

    trainer.recognizer.save_model()


if __name__ == "__main__":
    main()
