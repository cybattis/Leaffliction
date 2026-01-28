#!/usr/bin/env python3

import os

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Constants
IMAGE_SIZE = (224, 224)
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp'}


def load_model_and_labels(
    model_path: str
) -> Tuple[keras.Model, List[str], Dict[str, int], Optional[str]]:
    """
    Load a trained model and its class labels.

    Args:
        model_path: Path to .keras model file, or plant type ('apple'/'grape').

    Returns:
        Tuple of (model, class_names, class_indices, plant_type).
        plant_type is 'apple', 'grape', or None if unknown.
    """
    plant_type = None

    # Handle shortcut names
    if model_path.lower() in ['apple', 'grape']:
        plant_type = model_path.lower()
        model_dir = Path('models') / plant_type
        model_file = model_dir / f"{plant_type}_disease_model_best.keras"
        labels_file = model_dir / f"{plant_type}_disease_model_labels.json"
    else:
        model_file = Path(model_path)
        # Look for labels file in same directory
        model_dir = model_file.parent
        model_name = model_file.stem.replace('_best', '').replace('_final', '')
        labels_file = model_dir / f"{model_name}_labels.json"
        # Try to infer plant type from model path
        model_path_lower = str(model_file).lower()
        if 'apple' in model_path_lower:
            plant_type = 'apple'
        elif 'grape' in model_path_lower:
            plant_type = 'grape'

    # Check if model exists
    if not model_file.exists():
        raise FileNotFoundError(f"Model not found: {model_file}")

    print(f"Loading model from: {model_file}")
    model = keras.models.load_model(str(model_file))

    # Load labels
    if labels_file.exists():
        print(f"Loading labels from: {labels_file}")
        with open(labels_file, 'r') as f:
            labels_data = json.load(f)
        class_names = labels_data['class_names']
        class_indices = labels_data.get(
            'class_indices',
            {name: i for i, name in enumerate(class_names)}
        )
    else:
        print("Warning: Labels file not found, using default class indices")
        # Try to infer from model output shape
        num_classes = model.output_shape[-1]
        class_names = [f"Class_{i}" for i in range(num_classes)]
        class_indices = {name: i for i, name in enumerate(class_names)}

    print(f"Classes: {class_names}")
    if plant_type:
        print(f"Plant type: {plant_type.capitalize()}")
    return model, class_names, class_indices, plant_type


def preprocess_image(image_path: str) -> Optional[np.ndarray]:
    """
    Load and preprocess an image for prediction.

    Args:
        image_path: Path to the image file.

    Returns:
        Preprocessed image array ready for model input, or None if failed.
    """
    try:
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Warning: Could not load image: {image_path}")
            return None

        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Resize to model input size
        image = cv2.resize(image, IMAGE_SIZE)

        # Apply EfficientNet preprocessing
        image = tf.keras.applications.efficientnet.preprocess_input(image)

        # Add batch dimension
        image = np.expand_dims(image, axis=0)

        return image

    except Exception as e:
        print(f"Error preprocessing {image_path}: {e}")
        return None


def predict_single_image(
    model: keras.Model,
    image_path: str,
    class_names: List[str]
) -> Optional[Tuple[str, float, np.ndarray]]:
    """
    Predict the class of a single image.

    Args:
        model: Trained Keras model.
        image_path: Path to the image.
        class_names: List of class names.

    Returns:
        Tuple of (predicted_class, confidence, all_probabilities) or None.
    """
    # Preprocess image
    image = preprocess_image(image_path)
    if image is None:
        return None

    # Make prediction
    predictions = model.predict(image, verbose=0)
    probabilities = predictions[0]

    # Get predicted class
    predicted_idx = np.argmax(probabilities)
    predicted_class = class_names[predicted_idx]
    confidence = probabilities[predicted_idx]

    return predicted_class, float(confidence), probabilities


def scan_folder_for_images(
    folder_path: str,
    plant_type: Optional[str] = None
) -> Tuple[Dict[str, List[str]], List[str]]:
    """
    Scan a folder for images, organized by subdirectory (class).

    Args:
        folder_path: Path to the folder to scan.
        plant_type: If provided, only include folders starting with this plant
                    type (e.g., 'apple' matches 'Apple_*' folders).

    Returns:
        Tuple of:
        - Dictionary mapping class names to lists of image paths.
          If no subdirectories, returns {'unknown': [list of images]}.
        - List of skipped folder names (due to plant type filter).
    """
    folder = Path(folder_path)
    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder_path}")

    images_by_class = {}
    skipped_folders = []

    # Check if folder has subdirectories (class folders)
    subdirs = [d for d in folder.iterdir() if d.is_dir()]

    if subdirs:
        # Organized by class subdirectories
        for subdir in subdirs:
            class_name = subdir.name

            # Filter by plant type if specified
            if plant_type:
                # Check if folder starts with plant type (case-insensitive)
                if not class_name.lower().startswith(plant_type.lower()):
                    skipped_folders.append(class_name)
                    continue

            images = []
            for img_file in subdir.iterdir():
                if img_file.suffix.lower() in IMAGE_EXTENSIONS:
                    images.append(str(img_file))
            if images:
                images_by_class[class_name] = sorted(images)
    else:
        # Flat folder - no class labels
        images = []
        for img_file in folder.iterdir():
            if img_file.suffix.lower() in IMAGE_EXTENSIONS:
                images.append(str(img_file))
        if images:
            images_by_class['unknown'] = sorted(images)

    return images_by_class, skipped_folders


def predict_folder(
    model: keras.Model,
    folder_path: str,
    class_names: List[str],
    class_indices: Dict[str, int],
    plant_type: Optional[str] = None,
    verbose: bool = True
) -> Dict:
    """
    Run predictions on all images in a folder and calculate accuracy.

    Args:
        model: Trained Keras model.
        folder_path: Path to folder with images.
        class_names: List of class names.
        class_indices: Dictionary mapping class names to indices.
        plant_type: If provided, only predict on folders matching this plant.
        verbose: Whether to print per-image results.

    Returns:
        Dictionary with prediction results and metrics.
    """
    print(f"\nScanning folder: {folder_path}")
    images_by_class, skipped_folders = scan_folder_for_images(
        folder_path, plant_type
    )

    # Report skipped folders
    if skipped_folders:
        print(f"\nFiltering for plant type: {plant_type.capitalize()}")
        print(f"Skipped {len(skipped_folders)} non-{plant_type} folder(s):")
        for folder_name in skipped_folders:
            print(f"  - {folder_name}")

    if not images_by_class:
        print("No images found in folder.")
        return {}

    # Count total images
    total_images = sum(len(imgs) for imgs in images_by_class.values())
    print(f"Found {total_images} images in {len(images_by_class)} class(es)")

    # Check if we have ground truth labels
    has_ground_truth = 'unknown' not in images_by_class

    # Results storage
    results = {
        'predictions': [],
        'correct': 0,
        'incorrect': 0,
        'skipped': 0,
        'per_class': {name: {'correct': 0, 'total': 0} for name in class_names}
    }

    # Confusion matrix data
    y_true = []
    y_pred = []

    print("\n" + "=" * 70)
    print("Running predictions...")
    print("=" * 70)

    image_count = 0
    for true_class, image_paths in images_by_class.items():
        for image_path in image_paths:
            image_count += 1

            # Make prediction
            prediction = predict_single_image(model, image_path, class_names)

            if prediction is None:
                results['skipped'] += 1
                continue

            predicted_class, confidence, probabilities = prediction

            # Determine if correct (only if we have ground truth)
            is_correct = None
            if has_ground_truth and true_class in class_indices:
                is_correct = (predicted_class == true_class)
                if is_correct:
                    results['correct'] += 1
                else:
                    results['incorrect'] += 1

                # Update per-class stats
                if true_class in results['per_class']:
                    results['per_class'][true_class]['total'] += 1
                    if is_correct:
                        results['per_class'][true_class]['correct'] += 1

                # For confusion matrix
                y_true.append(class_indices[true_class])
                y_pred.append(class_indices[predicted_class])

            # Store result
            result_entry = {
                'image': image_path,
                'true_class': true_class if has_ground_truth else None,
                'predicted_class': predicted_class,
                'confidence': confidence,
                'correct': is_correct
            }
            results['predictions'].append(result_entry)

            # Print progress
            if verbose:
                status = ""
                if is_correct is not None:
                    status = "✓" if is_correct else "✗"

                # Calculate running accuracy
                evaluated = results['correct'] + results['incorrect']
                if evaluated > 0:
                    running_acc = results['correct'] / evaluated
                    acc_str = f" | Acc: {running_acc:.2%}"
                else:
                    acc_str = ""

                print(
                    f"[{image_count}/{total_images}]{acc_str} "
                    f"{Path(image_path).name}")
                print(f"    Predicted: {predicted_class} "
                      f"({confidence:.1%}) {status}")
                if not is_correct and is_correct is not None:
                    print(f"    True:      {true_class}")

    # Calculate metrics
    if has_ground_truth and (results['correct'] + results['incorrect']) > 0:
        total_evaluated = results['correct'] + results['incorrect']
        results['accuracy'] = results['correct'] / total_evaluated
        results['y_true'] = y_true
        results['y_pred'] = y_pred
    else:
        results['accuracy'] = None

    return results


def print_prediction_report(results: Dict, class_names: List[str]) -> None:
    """
    Print a summary report of predictions.

    Args:
        results: Results dictionary from predict_folder.
        class_names: List of class names.
    """
    print("\n" + "=" * 70)
    print("PREDICTION REPORT")
    print("=" * 70)

    total = len(results['predictions'])
    print(f"\nTotal images processed: {total}")
    print(f"Skipped (errors): {results['skipped']}")

    if results['accuracy'] is not None:
        print(f"\n{'='*40}")
        print(f"OVERALL ACCURACY: {results['accuracy']:.2%}")
        print(f"{'='*40}")
        print(f"  Correct:   {results['correct']}")
        print(f"  Incorrect: {results['incorrect']}")

        # Per-class breakdown
        print("\nPer-class accuracy:")
        print("-" * 40)
        for class_name in class_names:
            stats = results['per_class'].get(
                class_name, {'correct': 0, 'total': 0}
            )
            if stats['total'] > 0:
                acc = stats['correct'] / stats['total']
                print(f"  {class_name}:")
                print(f"    {stats['correct']}/{stats['total']} = {acc:.1%}")

        # Show misclassifications
        errors = [r for r in results['predictions'] if r['correct'] is False]
        if errors:
            print(f"\nMisclassified images ({len(errors)}):")
            print("-" * 40)
            for err in errors[:10]:  # Show first 10
                print(f"  {Path(err['image']).name}")
                print(f"    True: {err['true_class']}, "
                      f"Pred: {err['predicted_class']} "
                      f"({err['confidence']:.1%})")
            if len(errors) > 10:
                print(f"  ... and {len(errors) - 10} more")
    else:
        print("\nNo ground truth labels available (flat folder).")
        print("Showing predictions only:")
        print("-" * 40)
        for pred in results['predictions'][:20]:
            print(f"  {Path(pred['image']).name}: "
                  f"{pred['predicted_class']} ({pred['confidence']:.1%})")
        if len(results['predictions']) > 20:
            print(f"  ... and {len(results['predictions']) - 20} more")


def save_predictions_csv(results: Dict, output_path: str) -> None:
    """
    Save predictions to a CSV file.

    Args:
        results: Results dictionary from predict_folder.
        output_path: Path to save CSV file.
    """
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['image', 'true_class', 'predicted_class',
                         'confidence', 'correct'])
        for pred in results['predictions']:
            writer.writerow([
                pred['image'],
                pred['true_class'] or '',
                pred['predicted_class'],
                f"{pred['confidence']:.4f}",
                pred['correct'] if pred['correct'] is not None else ''
            ])

    print(f"\nPredictions saved to: {output_path}")


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Predict plant diseases from leaf images",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Predict on a folder using apple model
  python predict.py leaves/images -m apple

  # Predict on a folder using grape model
  python predict.py test_images/ -m grape

  # Predict using a specific model file
  python predict.py images/ -m models/apple/apple_disease_model_best.keras

  # Specify custom CSV filename
  python predict.py test_data/ -m apple --csv my_results.csv
        """
    )

    parser.add_argument(
        "input",
        type=str,
        help="Path to folder containing images"
    )

    parser.add_argument(
        "-m", "--model",
        type=str,
        required=True,
        help="Model to use: 'apple', 'grape', or path to .keras file"
    )

    parser.add_argument(
        "--csv",
        type=str,
        metavar="OUTPUT_FILE",
        help="Custom CSV filename (default: auto-generated)"
    )

    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Quiet mode - only show summary, not per-image results"
    )

    return parser.parse_args()


def main() -> int:
    """Main entry point."""
    args = parse_arguments()

    try:
        # Load model
        model, class_names, class_indices, plant_type = load_model_and_labels(
            args.model
        )

        input_path = Path(args.input)

        if not input_path.is_dir():
            print(f"Error: Input must be a directory: {input_path}")
            return 1

        # Folder prediction mode
        results = predict_folder(
            model,
            str(input_path),
            class_names,
            class_indices,
            plant_type=plant_type,
            verbose=not args.quiet
        )

        if results:
            print_prediction_report(results, class_names)

            # Auto-generate CSV filename if not specified
            if args.csv:
                csv_path = args.csv
            else:
                # Generate automatic CSV filename in predictions/ folder
                predictions_dir = Path("predictions")
                predictions_dir.mkdir(exist_ok=True)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                folder_name = input_path.name
                csv_filename = f"predictions_{folder_name}_{timestamp}.csv"
                csv_path = predictions_dir / csv_filename

            save_predictions_csv(results, str(csv_path))

        return 0

    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
