"""
Clean up annotations.json file by:
1. Removing duplicate entries (keeps last occurrence)
2. Fixing invalid labels (\, /, ., ..) by changing them to None
3. Normalizing paths
"""
import json
import shutil
from pathlib import Path
from datetime import datetime
import sys
import os

# Set UTF-8 encoding for console output (Windows compatibility)
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

sys.path.insert(0, str(Path(__file__).parent / "src"))
import config


def clean_annotations():
    """Clean duplicate entries and invalid labels from annotations.json"""

    annotation_file = Path(config.TRAINING_DIR) / "annotations.json"

    if not annotation_file.exists():
        print(f"\n❌ No annotations file found at: {annotation_file}")
        print("   Nothing to clean.\n")
        return

    # Create backup
    backup_file = annotation_file.with_suffix(f'.backup_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
    shutil.copy2(annotation_file, backup_file)
    print(f"\n📦 Backup created: {backup_file.name}")

    # Load annotations
    try:
        with open(annotation_file, 'r') as f:
            annotations = json.load(f)
    except Exception as e:
        print(f"\n❌ Error loading annotations: {e}")
        return

    print(f"\n📊 Original annotations:")
    print(f"   • {len(annotations)} total entries")

    # Count faces
    original_faces = sum(len(faces) for faces in annotations.values())
    print(f"   • {original_faces} total face annotations")

    # Track statistics
    cleaned_annotations = {}
    duplicate_image_count = 0
    duplicate_face_count = 0
    invalid_label_count = 0
    invalid_labels_found = []

    # Invalid labels to fix
    INVALID_LABELS = ['\\', '/', '.', '..', 'unknown', 'Unknown', 'UNKNOWN']

    # Process each entry
    for image_path, faces in annotations.items():
        # Normalize the path - if relative, resolve relative to DATA_DIR (parent of TRAINING_DIR)
        path_obj = Path(image_path)
        if not path_obj.is_absolute():
            # Paths like "training\file.jpg" are relative to DATA_DIR
            data_dir = Path(config.TRAINING_DIR).parent
            path_obj = (data_dir / path_obj).resolve()
        else:
            path_obj = path_obj.resolve()
        normalized_path = str(path_obj)

        if normalized_path in cleaned_annotations:
            duplicate_image_count += 1
            print(f"\n⚠️  Found duplicate path:")
            print(f"   Original: {image_path}")
            print(f"   Normalized: {normalized_path}")
            print(f"   Keeping the latest entry")

        # Clean face labels and remove duplicate boxes
        cleaned_faces = []
        seen_boxes = {}  # Track boxes we've seen (box tuple -> face dict)

        for face in faces:
            face_copy = face.copy()
            label = face_copy.get('label')
            box = face_copy.get('box')

            # Fix invalid labels
            if label in INVALID_LABELS:
                invalid_label_count += 1
                invalid_labels_found.append((Path(image_path).name, label))
                face_copy['label'] = None
                print(f"\n🔧 Fixed invalid label in {Path(image_path).name}: '{label}' → None")

            # Fix labels that are too short (likely errors)
            elif isinstance(label, str) and len(label.strip()) < 2 and label.strip() != '':
                invalid_label_count += 1
                invalid_labels_found.append((Path(image_path).name, label))
                face_copy['label'] = None
                print(f"\n🔧 Fixed short label in {Path(image_path).name}: '{label}' → None")

            # Check for duplicate boxes within this image
            if box:
                box_tuple = tuple(box)  # Convert list to tuple for dict key
                if box_tuple in seen_boxes:
                    duplicate_face_count += 1
                    print(f"\n🔧 Found duplicate face in {Path(image_path).name}: box {box} - keeping latest")
                # Store/overwrite with the latest version (last occurrence wins)
                seen_boxes[box_tuple] = face_copy
            else:
                # No box info, just add it
                cleaned_faces.append(face_copy)

        # Add all unique boxes (latest versions)
        cleaned_faces.extend(seen_boxes.values())

        # Store with normalized path (this will overwrite duplicates, keeping the last one)
        cleaned_annotations[normalized_path] = cleaned_faces

    # Convert back to original format (relative paths relative to PROJECT_ROOT)
    final_annotations = {}
    project_root = Path(config.PROJECT_ROOT).resolve()

    for norm_path, faces in cleaned_annotations.items():
        path_obj = Path(norm_path)

        # Try to make path relative to project root so paths are like "data/training/file.jpg"
        try:
            if path_obj.is_relative_to(project_root):
                rel_path = path_obj.relative_to(project_root)
                # Use forward slashes for consistency across platforms
                rel_path = str(rel_path).replace('\\', '/')
            else:
                # If not relative to project root, keep as absolute path
                rel_path = str(path_obj).replace('\\', '/')
        except:
            # Fallback to absolute path
            rel_path = str(path_obj).replace('\\', '/')

        final_annotations[rel_path] = faces

    # Calculate statistics
    cleaned_faces = sum(len(faces) for faces in final_annotations.values())

    print(f"\n✨ Cleaned annotations:")
    print(f"   • {len(final_annotations)} total image entries")
    print(f"   • {cleaned_faces} total face annotations")
    if duplicate_image_count > 0:
        print(f"   • Removed {duplicate_image_count} duplicate image entries")
    if duplicate_face_count > 0:
        print(f"   • Removed {duplicate_face_count} duplicate faces (same box in same image)")
    if invalid_label_count > 0:
        print(f"   • Fixed {invalid_label_count} invalid labels")

    # Check if anything changed
    needs_saving = (duplicate_image_count > 0 or
                    duplicate_face_count > 0 or
                    invalid_label_count > 0)

    if not needs_saving:
        print(f"\n✅ No issues found - annotations are already clean!")
        print(f"   Backup will be kept for safety: {backup_file.name}")
    else:
        # Save cleaned annotations
        try:
            with open(annotation_file, 'w') as f:
                json.dump(final_annotations, f, indent=2)

            print(f"\n✅ Cleaned annotations saved to: {annotation_file}")
            print(f"   Backup preserved at: {backup_file.name}")

            # Show what changed
            print(f"\n📈 Summary of changes:")
            total_changes = 0
            if duplicate_image_count > 0:
                print(f"   • Removed {duplicate_image_count} duplicate image entries")
                total_changes += duplicate_image_count
            if duplicate_face_count > 0:
                print(f"   • Removed {duplicate_face_count} duplicate faces (same box coordinates)")
                total_changes += duplicate_face_count
            if invalid_label_count > 0:
                print(f"   • Fixed {invalid_label_count} invalid labels (changed to None/unknown)")
                total_changes += invalid_label_count
            print(f"   • Total issues fixed: {total_changes}")

        except Exception as e:
            print(f"\n❌ Error saving cleaned annotations: {e}")
            print(f"   Original backup is safe at: {backup_file.name}")

    # Show labeled vs skipped breakdown
    labeled_faces = sum(
        sum(1 for face in faces if face.get('label'))
        for faces in final_annotations.values()
    )
    skipped_faces = sum(
        sum(1 for face in faces if face.get('label') is None)
        for faces in final_annotations.values()
    )

    print(f"\n👥 Face breakdown:")
    print(f"   • {labeled_faces} faces labeled")
    print(f"   • {skipped_faces} faces skipped/unknown")
    print(f"   • {cleaned_faces} total\n")


if __name__ == "__main__":
    try:
        print("\n" + "="*70)
        print(" "*20 + "ANNOTATIONS CLEANUP TOOL")
        print("="*70)
        clean_annotations()
        print("="*70 + "\n")
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
