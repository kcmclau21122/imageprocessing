"""
Training Data Analyzer
Analyzes your annotations.json to show which people need more photos
"""
import json
from pathlib import Path
from collections import Counter

def analyze_training_data(annotations_file='data/training/annotations.json'):
    """Analyze training data and provide recommendations."""
    
    print("\n" + "="*70)
    print(" "*15 + "TRAINING DATA ANALYSIS")
    print("="*70)
    
    # Load annotations
    try:
        with open(annotations_file, 'r') as f:
            annotations = json.load(f)
    except FileNotFoundError:
        print(f"❌ Annotations file not found: {annotations_file}")
        print("Run the data labeler first: python -m src.data_labeler")
        return
    
    # Analyze data
    total_images = len(annotations)
    total_faces = 0
    labeled_faces = 0
    person_counts = Counter()
    unlabeled_faces = 0
    
    for image_path, faces in annotations.items():
        total_faces += len(faces)
        for face in faces:
            label = face.get('label')
            if label:
                labeled_faces += 1
                person_counts[label] += 1
            else:
                unlabeled_faces += 1
    
    unique_people = len(person_counts)
    
    # Print overview
    print(f"\n📊 OVERVIEW")
    print(f"{'─'*70}")
    print(f"Total images processed: {total_images}")
    print(f"Total faces detected: {total_faces}")
    print(f"Labeled faces: {labeled_faces}")
    print(f"Unlabeled faces: {unlabeled_faces}")
    print(f"Unique people: {unique_people}")
    if labeled_faces > 0:
        print(f"Average faces per person: {labeled_faces / unique_people:.1f}")
    
    # Categorize people by sample count
    critical = []  # 1 face
    high_priority = []  # 2-4 faces
    good = []  # 5-9 faces
    excellent = []  # 10+ faces
    
    for person, count in person_counts.items():
        if count == 1:
            critical.append((person, count))
        elif count <= 4:
            high_priority.append((person, count))
        elif count <= 9:
            good.append((person, count))
        else:
            excellent.append((person, count))
    
    # Print detailed breakdown
    print(f"\n🚨 CRITICAL - Need 4-9 More Photos (Only 1 face)")
    print(f"{'─'*70}")
    if critical:
        for person, count in sorted(critical):
            print(f"  ⚠️  {person}: {count} face → Add 4-9 more photos")
        print(f"\nThese people WILL cause training issues!")
        print(f"Add more photos of them ASAP.")
    else:
        print("  ✓ None - Great!")
    
    print(f"\n⚠️  HIGH PRIORITY - Need 3-6 More Photos (2-4 faces)")
    print(f"{'─'*70}")
    if high_priority:
        for person, count in sorted(high_priority):
            needed = max(5 - count, 1)
            print(f"  • {person}: {count} faces → Add {needed}-{needed+3} more photos")
        print(f"\nThese people need more variety for good accuracy.")
    else:
        print("  ✓ None - Great!")
    
    print(f"\n👍 GOOD - Consider 1-5 More Photos (5-9 faces)")
    print(f"{'─'*70}")
    if good:
        for person, count in sorted(good):
            needed = max(10 - count, 1)
            print(f"  • {person}: {count} faces → Add {needed} more for excellent accuracy")
    else:
        print("  None in this category")
    
    print(f"\n✨ EXCELLENT - Well Represented (10+ faces)")
    print(f"{'─'*70}")
    if excellent:
        for person, count in sorted(excellent, key=lambda x: x[1], reverse=True):
            print(f"  ✓ {person}: {count} faces → Perfect!")
    else:
        print("  None yet - keep adding photos!")
    
    # Unlabeled faces reminder
    if unlabeled_faces > 0:
        print(f"\n💡 UNLABELED FACES")
        print(f"{'─'*70}")
        print(f"You have {unlabeled_faces} unlabeled faces!")
        print(f"These might include more samples of your {unique_people} people.")
        print(f"\nTo label them:")
        print(f"  python -m src.data_labeler")
    
    # Accuracy prediction
    print(f"\n🎯 ACCURACY PREDICTION")
    print(f"{'─'*70}")
    if critical:
        print(f"⚠️  Expected accuracy: 60-75%")
        print(f"   Reason: {len(critical)} people have only 1 face")
        print(f"   Action: Add more photos of critical people")
    elif high_priority:
        print(f"⚠️  Expected accuracy: 75-85%")
        print(f"   Reason: {len(high_priority)} people have 2-4 faces")
        print(f"   Action: Add more photos for 90%+ accuracy")
    elif good:
        print(f"✓ Expected accuracy: 85-90%")
        print(f"   Reason: Most people have 5-9 faces")
        print(f"   Action: Add a few more for guaranteed 90%+")
    else:
        print(f"✓ Expected accuracy: 90%+")
        print(f"   Reason: All people have 10+ faces - excellent!")
    
    # Recommendations
    print(f"\n📋 ACTION ITEMS")
    print(f"{'─'*70}")
    
    items = []
    if critical:
        items.append(f"1. URGENT: Add photos of {len(critical)} people with only 1 face")
    if high_priority:
        items.append(f"2. Add photos of {len(high_priority)} people with 2-4 faces")
    if unlabeled_faces > 0:
        items.append(f"3. Label {unlabeled_faces} unlabeled faces")
    if good:
        items.append(f"4. Optional: Add more photos of {len(good)} people for 90%+")
    
    if items:
        for item in items:
            print(f"  {item}")
    else:
        print("  ✓ Ready to train! Your data looks excellent.")
    
    # Training readiness
    print(f"\n🚀 READY TO TRAIN?")
    print(f"{'─'*70}")
    
    if critical:
        print("  ❌ NOT RECOMMENDED - Fix critical issues first")
        print("  Training will complete but accuracy will be low")
    elif high_priority and len(high_priority) > 5:
        print("  ⚠️  CAN TRAIN - But accuracy may be 75-85%")
        print("  Consider adding more photos for 90%+")
    elif labeled_faces < 100:
        print("  ⚠️  CAN TRAIN - But more data recommended")
        print("  Current: {labeled_faces} faces, Ideal: 150+ faces")
    else:
        print("  ✓ YES - Data looks good!")
        print("  Run: python train_model.py")
    
    print("="*70 + "\n")


if __name__ == "__main__":
    import sys
    
    # Allow custom path
    annotations_file = sys.argv[1] if len(sys.argv) > 1 else 'data/training/annotations.json'
    
    analyze_training_data(annotations_file)