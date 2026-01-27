#!/usr/bin/env python3

import argparse
import sys
import shutil
from pathlib import Path
from typing import Dict, List, Tuple
import random
import numpy as np
import matplotlib.pyplot as plt
import cv2

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp'}
TARGET_SIZE = (224, 224)


def flip_image(image: np.ndarray) -> np.ndarray:
    """Apply horizontal and vertical flip."""
    if random.random() > 0.5:
        image = cv2.flip(image, 1)
    if random.random() > 0.5:
        image = cv2.flip(image, 0)
    return image


def rotate_image(image: np.ndarray) -> np.ndarray:
    """Apply random rotation."""
    if random.random() > 0.5:
        angle = random.uniform(-90, -10)
    else:
        angle = random.uniform(10, 90)
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, rotation_matrix, (w, h),
                             borderMode=cv2.BORDER_CONSTANT,
                             borderValue=(0, 0, 0))
    return rotated


def blur_image(image: np.ndarray) -> np.ndarray:
    """Apply random blur effect."""
    kernel_size = random.choice([5, 7, 9, 11, 13])
    blurred = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    return blurred


def crop_image(image: np.ndarray) -> np.ndarray:
    """Apply random cropping and resize back to original size."""
    h, w = image.shape[:2]
    crop_factor = random.uniform(0.7, 0.95)
    crop_h = int(h * crop_factor)
    crop_w = int(w * crop_factor)
    start_h = random.randint(0, h - crop_h)
    start_w = random.randint(0, w - crop_w)
    cropped = image[start_h:start_h + crop_h, start_w:start_w + crop_w]
    resized = cv2.resize(cropped, (w, h))
    return resized


def zoom_image(image: np.ndarray) -> np.ndarray:
    """Apply random zoom in."""
    h, w = image.shape[:2]
    scale = random.uniform(1.2, 1.5)
    new_h, new_w = int(h * scale), int(w * scale)
    resized = cv2.resize(image, (new_w, new_h))
    start_h = (new_h - h) // 2
    start_w = (new_w - w) // 2
    result = resized[start_h:start_h + h, start_w:start_w + w]
    return result


def adjust_brightness(image: np.ndarray) -> np.ndarray:
    """Apply random brightness adjustment."""
    img_float = image.astype(np.float32)
    brightness = random.uniform(0.5, 2.0)
    brightened = img_float * brightness
    result = np.clip(brightened, 0, 255).astype(np.uint8)
    return result


def adjust_contrast(image: np.ndarray) -> np.ndarray:
    """Apply random contrast adjustment."""
    img_float = image.astype(np.float32)
    contrast = random.uniform(0.5, 2.0)
    mean = img_float.mean()
    contrasted = mean + contrast * (img_float - mean)
    result = np.clip(contrasted, 0, 255).astype(np.uint8)
    return result


def create_augmentation_functions() -> Dict[str, callable]:
    """
    Create dictionary of augmentation functions.

    Returns:
        Dictionary mapping augmentation names to functions.
    """
    return {
        "Flip": flip_image,
        "Rotate": rotate_image,
        "Zoom": zoom_image,
        "Brightness": adjust_brightness,
        "Contrast": adjust_contrast,
        "Blur": blur_image
    }


def load_and_preprocess_image(image_path: str) -> np.ndarray:
    """
    Load and preprocess an image for augmentation.

    Args:
        image_path: Path to the image file.

    Returns:
        Loaded image as numpy array (BGR format, 0-255 range).
    """
    try:
        # Load image using OpenCV (BGR format)
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")

        image = cv2.resize(image, TARGET_SIZE)

        return image

    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None


def scan_dataset_by_fruit(dataset_path: str,
                          fruit_type: str) -> Dict[str, List[str]]:
    """
    Scan dataset directory and organize images by class
    for a specific fruit type.

    Args:
        dataset_path: Path to dataset directory.
        fruit_type: Fruit type to filter by ('apple' or 'grape').

    Returns:
        Dictionary mapping class names to lists of image paths.
    """
    dataset = {}
    dataset_root = Path(dataset_path)

    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset path not found: {dataset_path}")

    # Convert fruit type to match directory naming
    fruit_prefix = fruit_type.capitalize()

    for class_dir in dataset_root.iterdir():
        if not class_dir.is_dir():
            continue

        class_name = class_dir.name

        # Filter by fruit type
        if not class_name.startswith(fruit_prefix):
            continue

        image_paths = []
        for image_file in class_dir.iterdir():
            if image_file.suffix.lower() in IMAGE_EXTENSIONS:
                image_paths.append(str(image_file))

        if image_paths:
            dataset[class_name] = image_paths
            print(f"Found {len(image_paths)} images in class '{class_name}'")

    return dataset


def apply_single_augmentation(
    image: np.ndarray,
    aug_name: str,
    aug_function: callable
) -> np.ndarray:
    """
    Apply a single augmentation function to an image.

    Args:
        image: Input image as numpy array (BGR format, 0-255 range).
        aug_name: Name of the augmentation for debugging.
        aug_function: Augmentation function to apply.

    Returns:
        Augmented image as numpy array (BGR format, 0-255 range).
    """
    try:
        augmented_image = aug_function(image.copy())
        if augmented_image is None:
            print(f"Warning: {aug_name} returned None, using original")
            return image.copy()

        if augmented_image.shape != image.shape:
            print(f"Warning: {aug_name} changed shape, using original")
            return image.copy()

        if augmented_image.dtype != np.uint8:
            augmented_image = np.clip(augmented_image, 0, 255).astype(np.uint8)

        print(f"  {aug_name}: range [{augmented_image.min()}, "
              f"{augmented_image.max()}]")

        return augmented_image

    except Exception as e:
        print(f"  Error with {aug_name}: {e}")
        return image.copy()


def calculate_augmentation_needs(
    dataset: Dict[str, List[str]]
) -> Tuple[int, Dict[str, int]]:
    """
    Calculate how many augmented images are needed per class.

    Args:
        dataset: Dictionary mapping class names to image paths.

    Returns:
        Tuple of (target_count, augmentation_counts_per_class).
    """
    class_counts = {cls: len(paths) for cls, paths in dataset.items()}
    target_count = max(class_counts.values())

    augmentation_needs = {}
    for class_name, current_count in class_counts.items():
        needed = target_count - current_count
        augmentation_needs[class_name] = max(0, needed)

    print("\nAugmentation Plan:")
    print(f"Target count per class: {target_count}")
    for class_name, needed in augmentation_needs.items():
        current = class_counts[class_name]
        print(f"  {class_name}: {current} -> {current + needed} "
              f"(+{needed} augmented)")

    return target_count, augmentation_needs


def generate_augmented_images(
    input_dir: str,
    output_dir: str,
    fruit_type: str,
    target_count: int = None
) -> None:
    """
    Generate augmented images to balance the dataset for a specific fruit type.

    Args:
        input_dir: Input directory with class subdirectories.
        output_dir: Output directory for augmented dataset.
        fruit_type: Fruit type to process ('apple' or 'grape').
        target_count: Target number of images per class (auto if None).
    """
    # Scan input dataset for specific fruit type
    dataset = scan_dataset_by_fruit(input_dir, fruit_type)

    if not dataset:
        print(f"No {fruit_type} image classes found in input directory.")
        return

    # Calculate augmentation needs
    if target_count is None:
        calc_result = calculate_augmentation_needs(dataset)
        target_count, augmentation_needs = calc_result
    else:
        augmentation_needs = {
            cls: max(0, target_count - len(paths))
            for cls, paths in dataset.items()
        }

    # Create augmentation functions
    aug_functions = create_augmentation_functions()
    print(f"\nAugmentation Functions: {list(aug_functions.keys())}")

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    total_generated = 0

    # Process each class
    for class_name, image_paths in dataset.items():
        class_output_dir = output_path / class_name
        class_output_dir.mkdir(exist_ok=True)

        needed_augmentations = augmentation_needs[class_name]

        if needed_augmentations == 0:
            print(f"\nClass '{class_name}': No augmentation needed")
            # Copy original images
            for i, img_path in enumerate(image_paths):
                src_path = Path(img_path)
                dst_name = f"original_{i:04d}{src_path.suffix}"
                dst_path = class_output_dir / dst_name
                shutil.copy2(src_path, dst_path)
            continue

        print(f"\nProcessing class '{class_name}': "
              f"generating {needed_augmentations} augmented images...")

        # Copy original images first
        for i, img_path in enumerate(image_paths):
            src_path = Path(img_path)
            dst_name = f"original_{i:04d}{src_path.suffix}"
            dst_path = class_output_dir / dst_name
            shutil.copy2(src_path, dst_path)

        # Generate augmented images using individual transformations
        augmented_count = 0
        source_index = 0

        while augmented_count < needed_augmentations:
            # Select source image (cycle through available images)
            source_path = image_paths[source_index % len(image_paths)]
            source_name = Path(source_path).stem

            # Load and preprocess image
            original_image = load_and_preprocess_image(source_path)
            if original_image is None:
                source_index += 1
                continue

            # Apply each augmentation type separately to the original image
            for aug_name, aug_function in aug_functions.items():
                if augmented_count >= needed_augmentations:
                    break

                try:
                    # Apply single augmentation to the original image
                    augmented = apply_single_augmentation(
                        original_image, aug_name, aug_function
                    )

                    # Generate output filename with transformation type
                    aug_filename = f"{source_name}_{aug_name}.jpg"
                    output_file = class_output_dir / aug_filename

                    # Save augmented image
                    cv2.imwrite(str(output_file), augmented)

                    augmented_count += 1
                    total_generated += 1

                    if augmented_count % 50 == 0:
                        print(f"  Generated {augmented_count}/"
                              f"{needed_augmentations} images")

                except Exception as e:
                    print(f"  Error generating {aug_name} augmentation: {e}")

            source_index += 1

        print(f"  Completed: {augmented_count} augmented images generated")

    print(f"\nAugmentation completed for {fruit_type}!")
    print(f"Total augmented images generated: {total_generated}")
    print(f"Balanced dataset saved to: {output_dir}")


def visualize_augmentations(image_path: str) -> None:
    """
    Create a visualization showing all augmentation types for a single image.

    Args:
        image_path: Path to the image to visualize.
    """
    # Load original image
    original_image = load_and_preprocess_image(image_path)
    if original_image is None:
        print(f"Could not load image: {image_path}")
        return

    # Create augmentation functions
    aug_functions = create_augmentation_functions()

    # Create figure for visualization (only augmented images, no original)
    num_augmentations = len(aug_functions)
    cols = 3
    rows = (num_augmentations + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
    axes = axes.flatten()

    # Apply and show each augmentation
    for i, (aug_name, aug_function) in enumerate(aug_functions.items()):
        augmented = apply_single_augmentation(
            original_image, aug_name, aug_function)

        # Convert BGR to RGB for matplotlib
        augmented_rgb = cv2.cvtColor(augmented, cv2.COLOR_BGR2RGB)
        axes[i].imshow(augmented_rgb)
        axes[i].set_title(f'{aug_name}', fontsize=12, fontweight='bold')
        axes[i].axis('off')

    # Hide unused subplots
    for i in range(num_augmentations, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()

    # Save visualization in visualization folder
    viz_dir = Path("visualization")
    viz_dir.mkdir(exist_ok=True)
    output_path = viz_dir / f"augmentation_output_{Path(image_path).stem}.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Augmentation visualization saved to: {output_path}")

    plt.show()


def parse_arguments() -> argparse.Namespace:
    """
    Parse command line arguments.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description="Data augmentation for Leaffliction project",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python augmentation.py apple leaves/images augmented_directory/
  python augmentation.py grape leaves/images augmented_directory/ \\
    --target-count 2000
  python augmentation.py --visualize leaves/images/Apple_healthy/image.JPG
        """
    )

    parser.add_argument(
        "--visualize",
        type=str,
        metavar="IMAGE_PATH",
        help="Visualize augmentations for a single image"
    )

    parser.add_argument(
        "fruit_type",
        type=str,
        nargs='?',
        choices=["apple", "grape"],
        help="Fruit type to process (apple or grape)"
    )

    parser.add_argument(
        "input_dir",
        type=str,
        nargs='?',
        help="Input directory containing class subdirectories with images"
    )

    parser.add_argument(
        "output_dir",
        type=str,
        nargs='?',
        help="Output directory for balanced dataset"
    )

    parser.add_argument(
        "--target-count",
        type=int,
        default=None,
        help="Target number of images per class (auto-detect if not specified)"
    )

    return parser.parse_args()


def main() -> int:
    """
    Main entry point for the augmentation script.

    Returns:
        Exit code (0 for success, non-zero for errors).
    """
    args = parse_arguments()

    try:
        # Visualization mode
        if args.visualize:
            print(f"Visualizing augmentations for: {args.visualize}")
            visualize_augmentations(args.visualize)
            return 0

        # Validate required arguments
        if not args.fruit_type or not args.input_dir or not args.output_dir:
            print("Error: fruit_type, input_dir and output_dir are "
                  "required for augmentation")
            print("Use --visualize IMAGE_PATH to visualize augmentations")
            return 1

        print(f"Fruit type: {args.fruit_type}")
        print(f"Input directory: {args.input_dir}")
        print(f"Output directory: {args.output_dir}")

        if args.target_count:
            print(f"Target count per class: {args.target_count}")
        else:
            print("Target count: Auto-detect (max class size)")

        # Generate augmented dataset
        generate_augmented_images(
            args.input_dir,
            args.output_dir,
            args.fruit_type,
            args.target_count
        )

        return 0

    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    # Set random seeds for reproducibility
    random.seed(42)
    np.random.seed(42)

    sys.exit(main())
