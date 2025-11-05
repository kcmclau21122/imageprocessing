"""
Example usage script demonstrating the Family Photo Identifier API
This shows how to use the system programmatically in your own scripts
"""
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

import config
from src.face_detection import FaceDetector
from src.face_recognition import FaceRecognizer
from test_model import FaceIdentifier


def example_1_detect_faces():
    """Example 1: Detect all faces in an image."""
    print("\n" + "="*60)
    print("EXAMPLE 1: Face Detection")
    print("="*60)
    
    # Initialize detector
    detector = FaceDetector(backend='mtcnn')
    
    # Detect faces in an image
    image_path = "path/to/your/image.jpg"  # Change this
    faces = detector.detect_faces(image_path)
    
    print(f"Detected {len(faces)} faces")
    
    for i, face in enumerate(faces):
        print(f"\nFace {i+1}:")
        print(f"  Position: {face['box']}")
        print(f"  Confidence: {face['confidence']:.2%}")


def example_2_train_model():
    """Example 2: Train a face recognition model."""
    print("\n" + "="*60)
    print("EXAMPLE 2: Training Model")
    print("="*60)
    
    from train_model import ModelTrainer
    
    # Initialize trainer
    trainer = ModelTrainer(model_name='Facenet512')
    
    # Train with custom settings
    results = trainer.train(
        classifier_type='svm',
        test_split=0.2
    )
    
    if results.get('success'):
        print(f"Training successful!")
        print(f"Test accuracy: {results.get('test_accuracy', 0):.2%}")
    else:
        print("Training failed")


def example_3_identify_single_image():
    """Example 3: Identify faces in a single image."""
    print("\n" + "="*60)
    print("EXAMPLE 3: Single Image Identification")
    print("="*60)
    
    # Initialize identifier (loads trained model)
    identifier = FaceIdentifier()
    
    # Identify faces
    image_path = "path/to/test/image.jpg"  # Change this
    results = identifier.identify_faces(image_path, confidence_threshold=0.75)
    
    print(f"Found {len(results)} faces:")
    
    for i, result in enumerate(results):
        print(f"\nFace {i+1}:")
        print(f"  Name: {result['predicted_label']}")
        print(f"  Confidence: {result['prediction_confidence']:.2%}")
        print(f"  Identified: {result['identified']}")


def example_4_batch_processing():
    """Example 4: Process multiple images."""
    print("\n" + "="*60)
    print("EXAMPLE 4: Batch Processing")
    print("="*60)
    
    from src.utils import get_image_files
    
    # Initialize identifier
    identifier = FaceIdentifier()
    
    # Get all images from directory
    image_dir = "path/to/your/photos"  # Change this
    image_files = get_image_files(image_dir)
    
    print(f"Processing {len(image_files)} images...")
    
    # Process in batch
    all_results = identifier.identify_batch(
        [str(f) for f in image_files],
        confidence_threshold=0.75
    )
    
    # Print summary
    total_faces = sum(len(faces) for faces in all_results.values())
    identified_faces = sum(
        sum(1 for f in faces if f['identified']) 
        for faces in all_results.values()
    )
    
    print(f"\nResults:")
    print(f"  Total faces: {total_faces}")
    print(f"  Identified: {identified_faces}")
    print(f"  Unknown: {total_faces - identified_faces}")


def example_5_custom_pipeline():
    """Example 5: Custom processing pipeline."""
    print("\n" + "="*60)
    print("EXAMPLE 5: Custom Pipeline")
    print("="*60)
    
    from src.face_detection import FaceDetector
    from src.face_recognition import FaceRecognizer
    from src.utils import load_image
    
    # Initialize components
    detector = FaceDetector(backend='mtcnn')
    recognizer = FaceRecognizer(model_name='Facenet512')
    
    # Load trained model
    recognizer.load_model()
    
    # Process image
    image_path = "path/to/image.jpg"  # Change this
    
    # Step 1: Detect faces
    faces = detector.detect_faces(image_path)
    print(f"Detected {len(faces)} faces")
    
    # Step 2: Identify each face
    for i, face in enumerate(faces):
        # Get embedding and predict
        label, confidence = recognizer.predict(face['face_img'])
        
        if confidence > 0.75:
            print(f"\nFace {i+1}: {label} ({confidence:.2%})")
        else:
            print(f"\nFace {i+1}: Unknown ({confidence:.2%})")


def example_6_with_llm_verification():
    """Example 6: Enhanced identification with LLM."""
    print("\n" + "="*60)
    print("EXAMPLE 6: LLM-Enhanced Identification")
    print("="*60)
    
    try:
        from src.llm_integration import EnhancedIdentifier
        from test_model import FaceIdentifier
        
        # Initialize enhanced identifier
        face_identifier = FaceIdentifier()
        enhanced = EnhancedIdentifier(face_identifier, llm_provider='ollama')
        
        # Identify with verification
        image_path = "path/to/image.jpg"  # Change this
        result = enhanced.identify_with_verification(image_path)
        
        # Print results
        print("\nFace Recognition Results:")
        for face in result['face_recognition']:
            print(f"  {face['predicted_label']}: {face['prediction_confidence']:.2%}")
        
        print("\nLLM Verification:")
        verification = result['llm_verification']
        if verification.get('success'):
            print(f"  {verification['verification']}")
        else:
            print(f"  Error: {verification.get('error')}")
            
    except ImportError:
        print("LLM integration requires Ollama or OpenAI setup")


def example_7_evaluate_accuracy():
    """Example 7: Evaluate model accuracy on test set."""
    print("\n" + "="*60)
    print("EXAMPLE 7: Model Evaluation")
    print("="*60)
    
    # Initialize identifier
    identifier = FaceIdentifier()
    
    # Run accuracy test on labeled test data
    results = identifier.test_accuracy(test_dir='data/testing')
    
    if results.get('success'):
        print(f"\nOverall Accuracy: {results['overall_accuracy']:.2%}")
        print(f"Correct: {results['correct_predictions']}/{results['total_faces']}")
        
        print("\nPer-Person Stats:")
        for person, stats in results['per_person_stats'].items():
            accuracy = stats['correct'] / stats['total']
            print(f"  {person}: {accuracy:.2%} ({stats['correct']}/{stats['total']})")


def example_8_export_results():
    """Example 8: Export results to structured format."""
    print("\n" + "="*60)
    print("EXAMPLE 8: Export Results")
    print("="*60)
    
    import json
    from src.utils import get_image_files
    
    # Initialize identifier
    identifier = FaceIdentifier()
    
    # Process images
    image_dir = "path/to/photos"  # Change this
    image_files = get_image_files(image_dir)
    
    results = identifier.identify_batch(
        [str(f) for f in image_files[:10]],  # Process first 10 for demo
        confidence_threshold=0.75
    )
    
    # Export to JSON
    output_file = "identification_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results exported to: {output_file}")
    
    # Could also export to CSV
    import pandas as pd
    
    # Flatten results for CSV
    rows = []
    for image_path, faces in results.items():
        for face in faces:
            rows.append({
                'image': Path(image_path).name,
                'person': face['predicted_label'],
                'confidence': face['prediction_confidence'],
                'identified': face['identified']
            })
    
    df = pd.DataFrame(rows)
    csv_file = "identification_results.csv"
    df.to_csv(csv_file, index=False)
    
    print(f"Results exported to: {csv_file}")


def main():
    """Run all examples (modify as needed)."""
    print("\n" + "="*70)
    print(" "*15 + "FAMILY PHOTO IDENTIFIER - EXAMPLES")
    print("="*70)
    
    print("\nNote: Update image paths in each example before running!")
    print("\nAvailable examples:")
    print("  1. Detect faces in an image")
    print("  2. Train a model")
    print("  3. Identify faces in single image")
    print("  4. Batch process multiple images")
    print("  5. Custom processing pipeline")
    print("  6. LLM-enhanced identification")
    print("  7. Evaluate model accuracy")
    print("  8. Export results to JSON/CSV")
    
    # Uncomment the examples you want to run:
    
    # example_1_detect_faces()
    # example_2_train_model()
    # example_3_identify_single_image()
    # example_4_batch_processing()
    # example_5_custom_pipeline()
    # example_6_with_llm_verification()
    # example_7_evaluate_accuracy()
    # example_8_export_results()
    
    print("\n" + "="*70)
    print("Uncomment the examples you want to run in this script!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()