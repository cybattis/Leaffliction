#!/usr/bin/env python3
"""
Image Transformation using PlantCV
Extracts 6 features from plant leaf images
"""
import os
import argparse
import matplotlib
import matplotlib.pyplot as plt
import cv2

from plantcv import plantcv as pcv

matplotlib.use('Agg')

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp'}


def apply_morphological_cleaning(mask, opening_size=5, closing_size=7):
    """
    Clean binary mask using morphological operations.

    Args:
        mask (ndarray): Binary mask (0 or 255)
        opening_size (int): Kernel size for opening operation (removes noise)
        closing_size (int): Kernel size for closing operation (fills holes)

    Returns:
        tuple: (opened_mask, closed_mask, final_clean_mask)
    """
    # Create structuring elements (kernels)
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                            (opening_size, opening_size))
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                             (closing_size, closing_size))

    # Step 1: Opening - Remove background noise (salt)
    opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)

    # Step 2: Closing - Fill holes inside leaf (pepper)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel_close)

    return opened, closed, closed  # Final mask = closed


def plant_mask(img, lower_green=None, upper_green=None):
    """Create a mask to isolate the plant from the background"""
    # Default HSV ranges for plant green vegetation
    if lower_green is None:
        # H=15° (yellow-green), S=40 (moderate), V=40 (dark)
        lower_green = (15, 40, 40)
    if upper_green is None:
        # H=85° (cyan-green), S=255 (vivid), V=255 (bright)
        upper_green = (80, 255, 255)

    blurred_img = pcv.gaussian_blur(img=img, ksize=(15, 15), sigma_x=0)
    bgr = cv2.cvtColor(blurred_img, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask, _ = pcv.threshold.custom_range(img=blurred_img,
                                         lower_thresh=lower_green,
                                         upper_thresh=upper_green,
                                         channel='HSV')

    _, _, clean_mask = apply_morphological_cleaning(mask,
                                                    opening_size=5,
                                                    closing_size=15)
    return hsv, mask, clean_mask, blurred_img


def gaussian_blur_transform(img):
    """Apply Gaussian blur to reduce noise"""
    gray_img = pcv.rgb2gray(img)
    threshold_dark = pcv.threshold.binary(gray_img=gray_img,
                                          threshold=100,
                                          object_type='dark')
    gaussian_img = pcv.gaussian_blur(img=threshold_dark,
                                     ksize=(5, 5),
                                     sigma_x=0)

    return gaussian_img


def mask_transform(img):
    """Create a mask to isolate the disease (non-green) parts of the leaf"""
    _, mask, _, _ = plant_mask(img, lower_green=(35, 40, 40))
    # Invert mask to get non-green (diseased) areas
    # mask = pcv.invert(mask)
    return mask


def object_analyse(img, mask):
    """Analyze object size and dimensions using the provided mask"""
    h, w = mask.shape[:2]
    roi = pcv.roi.rectangle(img=mask, x=0, y=0, h=h, w=w)

    # Filter objects (keep ones in ROI)
    # Using filter/quick_filter depending on pcv version.
    # Fallback/standard pattern:
    try:
        kept_mask = pcv.roi.filter(mask=mask, roi=roi, roi_type='partial')
    except AttributeError:
        # Fallback for older/different versions if needed, or use quick_filter
        kept_mask = pcv.roi.quick_filter(mask=mask, roi=roi)

    pcv.params.line_thickness = 2
    pcv.params.font = cv2.FONT_HERSHEY_SIMPLEX
    pcv.params.sample_label = ""
    analysis_image = pcv.analyze.size(img=img, labeled_mask=kept_mask)
    return analysis_image


def roi_contour_extraction(original_img, mask):
    """Extract and visualize ROI"""
    contours, _ = cv2.findContours(mask,
                                   cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    roi_contours = original_img.copy()
    cv2.drawContours(roi_contours, contours, -1, (0, 0, 255), 3)

    # Calculate metrics
    total_area = sum(cv2.contourArea(c) for c in contours)
    total_perimeter = sum(cv2.arcLength(c, True) for c in contours)

    shape_analysis = roi_contours.copy()
    cv2.putText(shape_analysis, f"Area: {int(total_area)} px²",
                (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(shape_analysis, f"Perimeter: {int(total_perimeter)} px",
                (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    return roi_contours, shape_analysis


def pseudolandmarks_transform(img, mask):
    """Extract pseudolandmarks from the object"""
    try:
        left, right, center_h = pcv.homology.x_axis_pseudolandmarks(img=img,
                                                                    mask=mask)
    except (RuntimeError, ValueError) as e:
        print(f"Warning: Pseudolandmarks failed: {e}")
        return img

    landmarks_img = img.copy()

    def draw_points(points, color):
        if points is not None:
            for pt in points:
                # Handle point format
                if len(pt) > 0 and hasattr(pt[0], '__len__'):
                    x, y = pt[0]
                else:
                    x, y = pt if len(pt) == 2 else (0, 0)
                cv2.circle(landmarks_img, (int(x), int(y)), 5, color, -1)

    draw_points(left, (255, 0, 0))
    draw_points(right, (0, 0, 255))
    draw_points(center_h, (0, 255, 0))

    return landmarks_img


def color_histogram_transform(img, mask, output_path):
    """
    Analyzes color and returns the histogram plot as an image array
    Includes RGB, HSV, and LAB color spaces on a single graph
    """
    # Convert to different color spaces
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)

    # Create figure with 1 subplot
    fig, ax = plt.subplots(figsize=(10, 8), dpi=60)

    # Calculate total non-zero pixels in mask for proportion
    total_pixels = cv2.countNonZero(mask) \
        if mask is not None \
        else img.shape[0] * img.shape[1]
    if total_pixels == 0:
        total_pixels = 1

    def plot_channels(image, labels, colors, linestyle='-'):
        for i, (label, color) in enumerate(zip(labels, colors)):
            hist = cv2.calcHist([image], [i], mask, [256], [0, 256])
            hist_prop = hist / total_pixels
            ax.plot(hist_prop, color=color, label=label, linestyle=linestyle)

    # Plot all spaces on the same axes
    plot_channels(rgb, ['Red', 'Green', 'Blue'], ['r', 'g', 'b'],
                  linestyle='-')
    plot_channels(hsv, ['Hue', 'Saturation', 'Value'],
                  ['orange', 'purple', 'brown'],
                  linestyle='--')
    plot_channels(lab, ['Lightness', 'A', 'B'],
                  ['black', 'cyan', 'magenta'],
                  linestyle=':')

    ax.set_title("Color Channels Proportion (RGB, HSV, LAB)")
    ax.set_xlabel("Pixel Intensity")
    ax.set_ylabel("Proportion")
    ax.set_xlim(0, 256)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    print(f"Saved result to {output_path}")
    plt.savefig(output_path)
    plt.close(fig)


def process_single_image(settings, image_path: str, output_path: str):
    """Process a single image and display all 6 transformations"""
    # Preprocessing - Mask background
    print(f"Processing image: {image_path}")

    original_img, _, _ = pcv.readimage(filename=image_path)
    if original_img is None:
        print(f"Error: Could not load image from {image_path}")
        return

    print(f"Image shape: {original_img.shape}")

    file, ext = os.path.splitext(os.path.basename(image_path))
    output_image_name = f"transformed_{file}{ext.lower()}"
    histo_img_name = f"transformed_{file}_histogram{ext.lower()}"
    mask_image_name = f"transformed_{file}_mask{ext.lower()}"

    output_path = os.path.join(os.path.dirname(output_path), output_image_name)
    histogram_path = os.path.join(os.path.dirname(output_path), histo_img_name)
    mask_path = os.path.join(os.path.dirname(output_path), mask_image_name)

    # Preprocessing - Mask background
    hsv, mask, clean_mask, blurred_img = plant_mask(original_img)

    # Transformations
    only_plant = pcv.apply_mask(img=original_img,
                                mask=clean_mask,
                                mask_color='black')

    if settings.mask:
        cv2.imwrite(mask_path, only_plant)
        print(f"Saved plant mask to {mask_path}")

    diseases_mask = mask_transform(only_plant)
    blur_img = gaussian_blur_transform(only_plant)
    obj_img = object_analyse(original_img, diseases_mask)
    roi_contours, roi_analyse = roi_contour_extraction(original_img, mask)
    landmark_img = pseudolandmarks_transform(original_img, mask)

    # Display images (convert BGR to RGB for matplotlib)
    if settings.debug:
        original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
        blurred_img = cv2.cvtColor(blurred_img, cv2.COLOR_BGR2RGB)
        opened_mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
        closed_mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
        only_plant = cv2.cvtColor(only_plant, cv2.COLOR_BGR2RGB)

        # Visualize the HSV channels and resulting mask
        fig, axes = plt.subplots(2, 5, figsize=(16, 8))
        # Row 1: Original, Blurred, HSV channels
        axes[0, 0].imshow(original_img)
        axes[0, 0].set_title('Original')
        axes[0, 0].axis('off')

        axes[0, 1].imshow(blurred_img)
        axes[0, 1].set_title('Blurred (k=15)')
        axes[0, 1].axis('off')

        axes[0, 2].imshow(hsv[:, :, 0], cmap='hsv')
        axes[0, 2].set_title('Hue Channel (Color)')
        axes[0, 2].axis('off')

        axes[0, 3].imshow(hsv[:, :, 1], cmap='gray')
        axes[0, 3].set_title('Saturation Channel')
        axes[0, 3].axis('off')

        # Row 2: Value channel, Green mask, Mask overlay, Comparison
        axes[0, 4].imshow(hsv[:, :, 2], cmap='gray')
        axes[0, 4].set_title('Value Channel (Brightness)')
        axes[0, 4].axis('off')

        axes[1, 0].imshow(opened_mask, cmap='gray')
        axes[1, 0].set_title('Open Mask')
        axes[1, 0].axis('off')

        axes[1, 1].imshow(closed_mask, cmap='gray')
        axes[1, 1].set_title('Closed Mask')
        axes[1, 1].axis('off')

        axes[1, 2].imshow(mask, cmap='gray')
        axes[1, 2].set_title('Clean Mask')
        axes[1, 2].axis('off')

        # Overlay mask on original
        overlay = original_img.copy()
        overlay[mask == 0] = overlay[mask == 0] // 2  # Darken background
        axes[1, 3].imshow(overlay)
        axes[1, 3].set_title('Mask Overlay')
        axes[1, 3].axis('off')

        axes[1, 4].imshow(only_plant)
        axes[1, 4].set_title('Only plant')
        axes[1, 4].axis('off')

        plt.tight_layout()
        plt.savefig(output_path)
        plt.close(fig)
    else:
        transformations = [
            (cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB), 'Original Image'),
            (cv2.cvtColor(clean_mask, cv2.COLOR_GRAY2RGB), 'Plant Mask'),
            (cv2.cvtColor(only_plant, cv2.COLOR_BGR2RGB), 'Preprocess image'),
            (cv2.cvtColor(blur_img, cv2.COLOR_BGR2RGB), 'Gaussian Blur'),
            (cv2.cvtColor(roi_contours, cv2.COLOR_RGB2BGR),
             'ROI Contour Extraction'),
            (cv2.cvtColor(roi_analyse, cv2.COLOR_BGR2RGB), 'ROI Analysis'),
            (cv2.cvtColor(obj_img, cv2.COLOR_BGR2RGB), 'Object Analysis'),
            (cv2.cvtColor(landmark_img, cv2.COLOR_BGR2RGB), 'Pseudolandmarks'),
        ]

        # Create figure with subplots
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        fig.suptitle(f'PlantCV Feature Extraction - '
                     f'{os.path.basename(image_path)}',
                     fontsize=16, fontweight='bold')

        for ax, (img_data, title) in zip(axes.flat, transformations):
            ax.imshow(img_data)
            ax.set_title(title, fontsize=12, fontweight='bold')
            ax.axis('off')

        plt.tight_layout()

        print(f"output_path: {output_path}")
        plt.savefig(output_path, dpi=60)
        plt.close(fig)

        if not settings.zip:
            print(f"Saved histogram result to {histogram_path}")
            color_histogram_transform(original_img, mask, histogram_path)

    print("Transformation complete!")


def main():
    parser = argparse.ArgumentParser(
        description='Image Transformation using PlantCV - '
                    'Extract 6 features from plant leaf images'
    )
    parser.add_argument('-i', '--image', type=str, metavar='PATH',
                        help='Path to input image')
    parser.add_argument('-src', '--source', type=str, metavar='DIR',
                        help='Path to source directory')
    parser.add_argument('-dst', '--destination', type=str, metavar='DIR',
                        help='Path to destination directory')
    parser.add_argument('-d', '--debug', action='store_true',
                        help='Enable debug mode')
    parser.add_argument('-m', '--mask', action='store_true',
                        help='Save binary mask')
    parser.add_argument('-z', '--zip', action='store_true',
                        help='Run in zip mode')
    args = parser.parse_args()

    if args.image is None and args.source is None:
        parser.print_help()
        return

    # Transform all images in a directory
    if args.source is not None:
        if args.image is not None:
            print("Error: Cannot process both an image and a directory.")
            exit(1)
        if not os.path.exists(args.source):
            print(f"Error: Source directory '{args.source}' not found!")
        if args.destination is None:
            print("Error: Destination directory must be "
                  "specified when processing a directory.")
            exit(1)
        if not os.path.exists(args.destination):
            os.makedirs(args.destination)
        for root, _, files in os.walk(args.source):
            for file in files:
                if any(file.lower().endswith(ext) for ext in IMAGE_EXTENSIONS):
                    input_path = str(os.path.join(root, file))
                    # Preserve subdirectory structure
                    rel_path = os.path.relpath(input_path, args.source)
                    output_path = str(os.path.join(args.destination, rel_path))
                    # Create subdirectory if it doesn't exist
                    os.makedirs(os.path.dirname(output_path), exist_ok=True)
                    process_single_image(args, input_path, output_path)

    # Process single image
    if args.image is not None:
        if args.destination is not None:
            print("Error: Destination directory must not "
                  "be specified when processing a single image.")
            exit(1)
        if not os.path.exists(args.image):
            print(f"Error: Image file '{args.image}' not found!")
            exit(1)

        # Create visualization directory and output path for single image
        viz_dir = "visualization"
        os.makedirs(viz_dir, exist_ok=True)

        # Create output path in visualization folder
        image_name = os.path.basename(args.image)
        output_path = os.path.join(viz_dir, image_name)

        print(f"Processing image: {args.image}")
        print(f"Output will be saved to: {viz_dir}/")
        process_single_image(args, args.image, output_path)


if __name__ == '__main__':
    print("PlantCV ", pcv.__version__)
    main()
