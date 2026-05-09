"""
Data Quality Checker for Face Recognition Dataset
Analyzes annotations.json to identify data quality issues and recommend improvements
"""
import json
import logging
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, List, Tuple
import sys

from src import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataQualityChecker:
    """Analyze and report data quality issues in the face recognition dataset."""

    def __init__(self, training_dir: str = None):
        self.training_dir = Path(training_dir or config.TRAINING_DIR)
        self.annotation_file = self.training_dir / "annotations.json"
        self.annotations = {}
        self.issues = defaultdict(list)
        self.stats = {}

    def load_annotations(self) -> bool:
        """Load annotations from file."""
        if not self.annotation_file.exists():
            logger.error(f"Annotations file not found: {self.annotation_file}")
            print(f"\n❌ No annotations file found at: {self.annotation_file}")
            print("   Please run data_labeler.py first to create annotations.\n")
            return False

        try:
            with open(self.annotation_file, 'r') as f:
                self.annotations = json.load(f)
            logger.info(f"Loaded {len(self.annotations)} annotated images")
            return True
        except Exception as e:
            logger.error(f"Error loading annotations: {e}")
            return False

    def analyze_labels(self) -> Dict[str, int]:
        """Analyze label distribution and identify issues."""
        label_counts = Counter()
        invalid_labels = []
        short_labels = []
        all_labels = []

        for image_path, faces in self.annotations.items():
            for face_info in faces:
                label = face_info.get('label')

                # Skip None, null, empty, or unknown labels (these are intentionally skipped/unlabeled faces)
                if label is None or label == 'None' or label == 'null':
                    continue

                label = str(label).strip()

                # Skip empty strings or unknown variations
                if not label or label.lower() in ['unknown', 'none', 'null', '']:
                    continue

                # Check for invalid labels
                if label in ['\\', '/', '.', '..']:
                    invalid_labels.append((image_path, label))
                    continue

                # Check for suspiciously short labels
                if len(label) < 2:
                    short_labels.append((image_path, label))
                    continue

                # Normalize label
                label = label.title()
                label_counts[label] += 1
                all_labels.append(label)

        # Store issues
        if invalid_labels:
            self.issues['invalid_labels'] = invalid_labels
        if short_labels:
            self.issues['short_labels'] = short_labels

        return label_counts

    def analyze_class_imbalance(self, label_counts: Dict[str, int]) -> List[Tuple[str, str]]:
        """Identify severely imbalanced classes."""
        if not label_counts:
            return []

        recommendations = []

        # Calculate statistics
        counts = list(label_counts.values())
        avg_count = sum(counts) / len(counts)
        max_count = max(counts)

        # Recommended minimum samples per person
        MIN_RECOMMENDED = 10
        CRITICAL_MIN = 5

        for person, count in sorted(label_counts.items(), key=lambda x: x[1]):
            if count < CRITICAL_MIN:
                recommendations.append((person, 'critical', count))
            elif count < MIN_RECOMMENDED:
                recommendations.append((person, 'warning', count))

        return recommendations

    def check_duplicate_names(self, label_counts: Dict[str, int]) -> List[List[str]]:
        """Find potentially duplicate person names (case variations, typos)."""
        duplicates = []
        names = list(label_counts.keys())

        for i, name1 in enumerate(names):
            similar = [name1]
            for name2 in names[i+1:]:
                # Check for case-insensitive matches
                if name1.lower() == name2.lower() and name1 != name2:
                    similar.append(name2)

            if len(similar) > 1 and similar not in duplicates:
                duplicates.append(similar)

        return duplicates

    def calculate_statistics(self, label_counts: Dict[str, int]) -> Dict:
        """Calculate overall dataset statistics."""
        total_images = len(self.annotations)
        total_faces = sum(len(faces) for faces in self.annotations.values())
        labeled_faces = sum(label_counts.values())
        unlabeled_faces = total_faces - labeled_faces
        unique_people = len(label_counts)

        avg_samples = labeled_faces / unique_people if unique_people > 0 else 0

        return {
            'total_images': total_images,
            'total_faces': total_faces,
            'labeled_faces': labeled_faces,
            'unlabeled_faces': unlabeled_faces,
            'unique_people': unique_people,
            'avg_samples_per_person': avg_samples,
        }

    def generate_report(self):
        """Generate comprehensive data quality report."""
        print("\n" + "=" * 80)
        print(" " * 25 + "DATA QUALITY REPORT")
        print("=" * 80)

        # Load and analyze data
        if not self.load_annotations():
            return

        label_counts = self.analyze_labels()
        self.stats = self.calculate_statistics(label_counts)
        imbalance_issues = self.analyze_class_imbalance(label_counts)
        duplicate_names = self.check_duplicate_names(label_counts)

        # Print statistics
        print("\n📊 DATASET STATISTICS")
        print("-" * 80)
        print(f"  Total images processed:     {self.stats['total_images']}")
        print(f"  Total faces detected:       {self.stats['total_faces']}")
        print(f"  Labeled faces:              {self.stats['labeled_faces']}")
        print(f"  Unlabeled faces:            {self.stats['unlabeled_faces']}")
        print(f"  Unique people:              {self.stats['unique_people']}")
        print(f"  Avg samples per person:     {self.stats['avg_samples_per_person']:.1f}")

        # Overall assessment
        print("\n📈 OVERALL ASSESSMENT")
        print("-" * 80)

        issues_found = 0

        # Check for minimum data
        if self.stats['unique_people'] < 2:
            print("  ❌ CRITICAL: Need at least 2 people for classification")
            issues_found += 1
        elif self.stats['unique_people'] < 5:
            print("  ⚠️  WARNING: Limited number of people - consider adding more diversity")
            issues_found += 1
        else:
            print(f"  ✅ Good: {self.stats['unique_people']} unique people identified")

        if self.stats['avg_samples_per_person'] < 5:
            print(f"  ❌ CRITICAL: Average {self.stats['avg_samples_per_person']:.1f} samples per person (need 10+)")
            issues_found += 1
        elif self.stats['avg_samples_per_person'] < 10:
            print(f"  ⚠️  WARNING: Average {self.stats['avg_samples_per_person']:.1f} samples per person (recommend 10+)")
            issues_found += 1
        else:
            print(f"  ✅ Good: Average {self.stats['avg_samples_per_person']:.1f} samples per person")

        # Invalid labels
        if 'invalid_labels' in self.issues:
            print(f"\n❌ INVALID LABELS FOUND: {len(self.issues['invalid_labels'])}")
            print("-" * 80)
            for img_path, label in self.issues['invalid_labels'][:5]:
                print(f"  • {Path(img_path).name}: '{label}'")
            if len(self.issues['invalid_labels']) > 5:
                print(f"  ... and {len(self.issues['invalid_labels']) - 5} more")
            print("\n  Action: Re-label these faces with valid person names")
            issues_found += len(self.issues['invalid_labels'])

        # Short labels
        if 'short_labels' in self.issues:
            print(f"\n⚠️  SUSPICIOUSLY SHORT LABELS: {len(self.issues['short_labels'])}")
            print("-" * 80)
            for img_path, label in self.issues['short_labels'][:5]:
                print(f"  • {Path(img_path).name}: '{label}'")
            if len(self.issues['short_labels']) > 5:
                print(f"  ... and {len(self.issues['short_labels']) - 5} more")
            print("\n  Action: Verify these labels are correct")
            issues_found += len(self.issues['short_labels'])

        # Duplicate names
        if duplicate_names:
            print(f"\n⚠️  POTENTIAL DUPLICATE NAMES: {len(duplicate_names)} groups")
            print("-" * 80)
            for group in duplicate_names:
                print(f"  • Similar names: {', '.join(group)}")
            print("\n  Action: Consolidate these names if they refer to the same person")
            issues_found += len(duplicate_names)

        # Class imbalance
        if imbalance_issues:
            print(f"\n⚠️  CLASS IMBALANCE DETECTED: {len(imbalance_issues)} people need more samples")
            print("-" * 80)

            critical = [p for p, severity, count in imbalance_issues if severity == 'critical']
            warning = [p for p, severity, count in imbalance_issues if severity == 'warning']

            if critical:
                print(f"\n  🔴 CRITICAL (< 5 samples): {len(critical)} people")
                for person, severity, count in imbalance_issues:
                    if severity == 'critical':
                        print(f"     • {person}: {count} samples (need 5+ immediately)")

            if warning:
                print(f"\n  🟡 WARNING (< 10 samples): {len(warning)} people")
                for person, severity, count in imbalance_issues:
                    if severity == 'warning':
                        print(f"     • {person}: {count} samples (recommend 10+ for good accuracy)")

            print("\n  Action: Add more photos containing these people")
            issues_found += len(imbalance_issues)

        # Class distribution
        print(f"\n📋 CLASS DISTRIBUTION (Top 20)")
        print("-" * 80)
        for person, count in sorted(label_counts.items(), key=lambda x: x[1], reverse=True)[:20]:
            bar_length = min(50, count)
            bar = "█" * bar_length
            status = "✅" if count >= 10 else "⚠️" if count >= 5 else "❌"
            print(f"  {status} {person:20s} {count:3d} {bar}")

        if len(label_counts) > 20:
            print(f"\n  ... and {len(label_counts) - 20} more people")

        # Recommendations
        print(f"\n💡 RECOMMENDATIONS")
        print("-" * 80)

        if issues_found == 0:
            print("  ✅ Your dataset looks good! No critical issues found.")
            print("  ✅ You should be able to achieve good recognition accuracy.")
        else:
            print(f"  Found {issues_found} issues that may impact model accuracy:\n")

            priority_actions = []

            if 'invalid_labels' in self.issues:
                priority_actions.append("1. IMMEDIATE: Remove or fix invalid labels (\\, /, etc.)")

            if imbalance_issues:
                critical_count = len([p for p, s, c in imbalance_issues if s == 'critical'])
                if critical_count > 0:
                    priority_actions.append(f"2. HIGH PRIORITY: Add photos for {critical_count} people with <5 samples")

            if duplicate_names:
                priority_actions.append(f"3. MEDIUM: Consolidate duplicate names")

            if self.stats['avg_samples_per_person'] < 10:
                priority_actions.append("4. RECOMMENDED: Aim for 10-15 samples per person overall")

            for action in priority_actions:
                print(f"  • {action}")

            print("\n  To fix issues:")
            print("    - Run: python src/data_labeler.py")
            print("    - Or manually edit: " + str(self.annotation_file))

        print("\n" + "=" * 80)

        # Return status code
        return 0 if issues_found == 0 else 1


def main():
    """Main entry point."""
    checker = DataQualityChecker()
    exit_code = checker.generate_report()

    print("\n💾 Tip: After fixing issues, re-run this script to verify improvements")
    print("🚀 Then run: python train_model.py --model ArcFace --cross-validate\n")

    sys.exit(exit_code)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
