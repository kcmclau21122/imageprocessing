"""
Face recognition module using embeddings and classification
Supports multiple embedding models (Facenet, ArcFace, VGG-Face, etc.)
"""
import logging
import pickle
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from deepface import DeepFace
import torch

import config

logger = logging.getLogger(__name__)


class FaceRecognizer:
    """
    Face recognition system using embeddings and machine learning classifiers.
    """
    
    def __init__(self, model_name: str = None):
        """
        Initialize face recognizer.
        
        Args:
            model_name: Embedding model name ('VGG-Face', 'Facenet', 'Facenet512', 
                       'OpenFace', 'DeepFace', 'ArcFace')
        """
        self.model_name = model_name or config.EMBEDDING_MODEL
        self.embeddings = []
        self.labels = []
        self.label_encoder = LabelEncoder()
        self.classifier = None
        
        logger.info(f"Initialized FaceRecognizer with {self.model_name} model")
    
    def extract_embedding(self, face_img: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract face embedding from a face image.
        
        Args:
            face_img: Face image as numpy array (RGB)
            
        Returns:
            Embedding vector as numpy array (or None if extraction fails)
        """
        try:
            # Ensure image is in correct format
            if face_img.dtype != np.uint8:
                face_img = (face_img * 255).astype(np.uint8)
            
            # Extract embedding using DeepFace
            embedding_objs = DeepFace.represent(
                img_path=face_img,
                model_name=self.model_name,
                enforce_detection=False
            )
            
            if embedding_objs:
                embedding = np.array(embedding_objs[0]["embedding"])
                return embedding
            
            return None
            
        except Exception as e:
            logger.error(f"Error extracting embedding: {str(e)}")
            return None
    
    def add_face(self, face_img: np.ndarray, label: str) -> bool:
        """
        Add a face and its label to the training set.
        
        Args:
            face_img: Face image as numpy array
            label: Person's name/label
            
        Returns:
            True if successful, False otherwise
        """
        try:
            embedding = self.extract_embedding(face_img)
            
            if embedding is not None:
                self.embeddings.append(embedding)
                self.labels.append(label)
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error adding face: {str(e)}")
            return False
    
    def train(self, classifier_type: str = 'svm') -> bool:
        """
        Train the classifier on collected embeddings.
        
        Args:
            classifier_type: Type of classifier ('svm', 'knn', 'random_forest')
            
        Returns:
            True if training successful, False otherwise
        """
        try:
            if len(self.embeddings) == 0:
                logger.error("No embeddings to train on")
                return False
            
            # Convert to numpy arrays
            X = np.array(self.embeddings)
            y = np.array(self.labels)
            
            # Encode labels
            y_encoded = self.label_encoder.fit_transform(y)
            
            # Initialize classifier
            if classifier_type == 'svm':
                self.classifier = SVC(kernel='linear', probability=True, C=1.0)
            elif classifier_type == 'knn':
                self.classifier = KNeighborsClassifier(n_neighbors=5, weights='distance')
            elif classifier_type == 'random_forest':
                self.classifier = RandomForestClassifier(n_estimators=100, random_state=42)
            else:
                raise ValueError(f"Unknown classifier type: {classifier_type}")
            
            # Train classifier
            self.classifier.fit(X, y_encoded)
            
            logger.info(f"Trained {classifier_type} classifier on {len(X)} samples with {len(set(y))} classes")
            return True
            
        except Exception as e:
            logger.error(f"Error training classifier: {str(e)}")
            return False
    
    def predict(self, face_img: np.ndarray, return_confidence: bool = True) -> Tuple[Optional[str], float]:
        """
        Predict the identity of a face.
        
        Args:
            face_img: Face image as numpy array
            return_confidence: Whether to return confidence score
            
        Returns:
            Tuple of (predicted_label, confidence) or (None, 0.0) if prediction fails
        """
        try:
            if self.classifier is None:
                logger.error("Classifier not trained")
                return None, 0.0
            
            # Extract embedding
            embedding = self.extract_embedding(face_img)
            
            if embedding is None:
                return None, 0.0
            
            # Predict
            embedding = embedding.reshape(1, -1)
            prediction = self.classifier.predict(embedding)
            
            # Get confidence
            if hasattr(self.classifier, 'predict_proba'):
                proba = self.classifier.predict_proba(embedding)
                confidence = np.max(proba)
            else:
                # For classifiers without predict_proba
                decision = self.classifier.decision_function(embedding)
                confidence = np.max(decision) / np.sum(np.abs(decision))
            
            # Decode label
            label = self.label_encoder.inverse_transform(prediction)[0]
            
            return label, confidence
            
        except Exception as e:
            logger.error(f"Error predicting face: {str(e)}")
            return None, 0.0
    
    def predict_batch(self, face_images: List[np.ndarray]) -> List[Tuple[Optional[str], float]]:
        """
        Predict identities for multiple faces.
        
        Args:
            face_images: List of face images
            
        Returns:
            List of (label, confidence) tuples
        """
        results = []
        
        for face_img in face_images:
            label, confidence = self.predict(face_img)
            results.append((label, confidence))
        
        return results
    
    def evaluate(self, test_embeddings: List[np.ndarray], test_labels: List[str]) -> Dict:
        """
        Evaluate classifier performance on test data.
        
        Args:
            test_embeddings: List of test embeddings
            test_labels: List of true labels
            
        Returns:
            Dictionary with evaluation metrics
        """
        try:
            if self.classifier is None:
                logger.error("Classifier not trained")
                return {}
            
            X_test = np.array(test_embeddings)
            y_test = self.label_encoder.transform(test_labels)
            
            # Predict
            y_pred = self.classifier.predict(X_test)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            report = classification_report(
                y_test, y_pred,
                target_names=self.label_encoder.classes_,
                output_dict=True
            )
            
            logger.info(f"Evaluation accuracy: {accuracy:.2%}")
            
            return {
                'accuracy': accuracy,
                'classification_report': report
            }
            
        except Exception as e:
            logger.error(f"Error evaluating classifier: {str(e)}")
            return {}
    
    def save_model(self, embeddings_path: str = None, labels_path: str = None, 
                   classifier_path: str = None):
        """
        Save the trained model to disk.
        
        Args:
            embeddings_path: Path to save embeddings
            labels_path: Path to save labels
            classifier_path: Path to save classifier
        """
        try:
            embeddings_path = embeddings_path or config.EMBEDDINGS_FILE
            labels_path = labels_path or config.LABELS_FILE
            classifier_path = classifier_path or config.CLASSIFIER_FILE
            
            # Save embeddings
            with open(embeddings_path, 'wb') as f:
                pickle.dump({
                    'embeddings': self.embeddings,
                    'labels': self.labels,
                    'model_name': self.model_name
                }, f)
            
            # Save label encoder and classifier
            with open(classifier_path, 'wb') as f:
                pickle.dump({
                    'classifier': self.classifier,
                    'label_encoder': self.label_encoder
                }, f)
            
            # Save labels as JSON for easy reading
            label_info = {
                'classes': self.label_encoder.classes_.tolist(),
                'num_samples': len(self.labels),
                'num_classes': len(set(self.labels))
            }
            
            with open(labels_path, 'w') as f:
                json.dump(label_info, f, indent=2)
            
            logger.info(f"Model saved to {embeddings_path}, {labels_path}, {classifier_path}")
            
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
    
    def load_model(self, embeddings_path: str = None, labels_path: str = None,
                   classifier_path: str = None):
        """
        Load a trained model from disk.
        
        Args:
            embeddings_path: Path to embeddings file
            labels_path: Path to labels file
            classifier_path: Path to classifier file
        """
        try:
            embeddings_path = embeddings_path or config.EMBEDDINGS_FILE
            classifier_path = classifier_path or config.CLASSIFIER_FILE
            
            # Load embeddings
            with open(embeddings_path, 'rb') as f:
                data = pickle.load(f)
                self.embeddings = data['embeddings']
                self.labels = data['labels']
                self.model_name = data['model_name']
            
            # Load classifier
            with open(classifier_path, 'rb') as f:
                data = pickle.load(f)
                self.classifier = data['classifier']
                self.label_encoder = data['label_encoder']
            
            logger.info(f"Model loaded from {embeddings_path} and {classifier_path}")
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")