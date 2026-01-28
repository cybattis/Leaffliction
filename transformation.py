#!/usr/bin/env python3
"""
Image Transformation using PlantCV
Extracts 6 features from plant leaf images
"""
import os
import argparse
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cv2
import numpy as np

# Import libraries
from plantcv import plantcv as pcv


def gaussian_blur_transform(img):
    """Apply Gaussian blur to reduce noise"""
    gray_img = pcv.rgb2gray(img)
    threshold_dark = pcv.threshold.binary(gray_img=gray_img, threshold=100, object_type='dark')
    gaussian_img = pcv.gaussian_blur(img=threshold_dark, ksize=(5, 5), sigma_x=0)
    return gaussian_img


def mask_transform(img):
    """Create a mask to isolate the disease (non-green) parts of the leaf"""
    # Isolate healthy green parts using 'a' channel (Green-Magenta axis)
    a = pcv.rgb2gray_lab(rgb_img=img, channel='a')
    mask_green = pcv.threshold.binary(gray_img=a, threshold=108, object_type='dark')

    # Combine: Disease = (Plant) AND (NOT Green)
    mask_not_green = pcv.invert(gray_img=mask_green)
    mask_fill = pcv.fill(bin_img=mask_not_green, size=25)

    # Apply mask to show only disease spots on black background
    masked = pcv.apply_mask(img=img, mask=mask_fill, mask_color='white')

    return masked


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

    analysis_image = pcv.analyze.size(img=img, labeled_mask=kept_mask)
    return analysis_image


def roi_extraction(img, mask):
    """Extract and visualize ROI"""
    # Create a bounding box around the leaf
    x, y, w, h = cv2.boundingRect(mask)
    roi_img = img.copy()
    cv2.rectangle(roi_img, (x, y), (x+w, y+h), (0, 255, 0), 5)
    return roi_img


def pseudolandmarks_transform(img, mask):
    """Extract pseudolandmarks from the object"""
    try:
        left, right, center_h = pcv.homology.x_axis_pseudolandmarks(img=img, mask=mask)
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
                    x, y = pt if len(pt) == 2 else (0,0)
                cv2.circle(landmarks_img, (int(x), int(y)), 5, color, -1)

    draw_points(left, (255, 0, 0))
    draw_points(right, (0, 0, 255))
    draw_points(center_h, (0, 255, 0))

    return landmarks_img


def color_histogram_transform(img, mask):
    """
    Analyzes color and returns the histogram plot as an image array
    Includes RGB, HSV, and LAB color spaces using plantcv.visualize.histogram
    """
    # 1. RGB Histogram (Original Image)
    # pcv.visualize.histogram plots pixel counts for channels
    hist_rgb = pcv.visualize.histogram(img=img, mask=mask, title="RGB Histogram")
    filename_rgb = "temp_hist_rgb.png"
    hist_rgb.save(filename_rgb)

    # 2. HSV Histogram (Hue, Saturation, Value)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hist_hsv = pcv.visualize.histogram(img=hsv, mask=mask, title="HSV Histogram")
    filename_hsv = "temp_hist_hsv.png"
    hist_hsv.save(filename_hsv)

    # 3. LAB Histogram (Lightness, Green-Magenta, Blue-Yellow)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
    hist_lab = pcv.visualize.histogram(img=lab, mask=mask, title="LAB Histogram")
    filename_lab = "temp_hist_lab.png"
    hist_lab.save(filename_lab)

    # Read the generated plots
    img_rgb_plot = cv2.imread(filename_rgb)
    img_hsv_plot = cv2.imread(filename_hsv)
    img_lab_plot = cv2.imread(filename_lab)

    # Clean up temporary files
    for f in [filename_rgb, filename_hsv, filename_lab]:
        if os.path.exists(f):
            os.remove(f)

    # Helper to resize images to match width of the first image
    def match_width(target_img, source_img):
        if source_img is None: return target_img
        h_t, w_t = target_img.shape[:2]
        h_s, w_s = source_img.shape[:2]
        if w_t != w_s:
            scale = w_t / w_s
            return cv2.resize(source_img, (w_t, int(h_s * scale)))
        return source_img

    # Resize plots to match RGB plot width
    img_hsv_plot = match_width(img_rgb_plot, img_hsv_plot)
    img_lab_plot = match_width(img_rgb_plot, img_lab_plot)

    # Stack vertically: RGB -> HSV -> LAB
    final_hist_img = cv2.vconcat([img_rgb_plot, img_hsv_plot, img_lab_plot])

    return final_hist_img


def plant_mask(img):
    """Create a mask to isolate the plant from the background"""
    b_channel = pcv.rgb2gray_hsv(rgb_img=img, channel='s')
    # Use automatic triangle thresholding instead of fixed threshold
    b_thresh = pcv.threshold.triangle(gray_img=b_channel, object_type='light', xstep=10)
    b_clean = pcv.fill(bin_img=b_thresh, size=500)

    # Fill holes
    b_inv = pcv.invert(gray_img=b_clean)
    b_filled_inv = pcv.fill(bin_img=b_inv, size=1000)
    b_filled = pcv.invert(gray_img=b_filled_inv)

    smooth_edge = pcv.median_blur(gray_img=b_filled, ksize=5)
    return smooth_edge


def process_single_image(image_path, output_path):
    """Process a single image and display all 6 transformations"""
    # Preprocessing - Mask background
    img, _, _ = pcv.readimage(filename=image_path)
    if img is None:
        print(f"Error: Could not load image from {image_path}")
        return

    print(f"Processing image: {image_path}")
    print(f"Image shape: {img.shape}")

    mask = plant_mask(img)
    only_plant = pcv.apply_mask(img=img, mask=mask, mask_color='white')

    # Apply all transformations
    blur_img = gaussian_blur_transform(only_plant)
    masked_img = mask_transform(only_plant)

    # Run newly implemented transformations
    obj_img = object_analyse(img, mask)
    roi_img = roi_extraction(img, mask)
    landmark_img = pseudolandmarks_transform(img, mask)
    hist_img = color_histogram_transform(img, mask)

    # Create figure with subplots
    fig, axes = plt.subplots(3, 3, figsize=(20, 15))
    fig.suptitle(f'PlantCV Feature Extraction - {os.path.basename(image_path)}', 
                 fontsize=16, fontweight='bold')

    # Display images (convert BGR to RGB for matplotlib)
    # mask is grayscale, others are BGR
    transformations = [
        (cv2.cvtColor(img, cv2.COLOR_BGR2RGB), 'Original Image'),
        (cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB), 'Plant Mask'),
        (cv2.cvtColor(only_plant, cv2.COLOR_BGR2RGB), 'Preprocess image'),
        (cv2.cvtColor(blur_img, cv2.COLOR_BGR2RGB), 'Gaussian Blur'),
        (cv2.cvtColor(masked_img, cv2.COLOR_BGR2RGB), 'Disease Masking'),
        (cv2.cvtColor(roi_img, cv2.COLOR_BGR2RGB), 'ROI Extraction'),
        (cv2.cvtColor(obj_img, cv2.COLOR_BGR2RGB), 'Object Analysis'),
        (cv2.cvtColor(landmark_img, cv2.COLOR_BGR2RGB), 'Pseudolandmarks'),
        (cv2.cvtColor(hist_img, cv2.COLOR_BGR2RGB), 'Color Histogram')
    ]

    for ax, (img_data, title) in zip(axes.flat, transformations):
        ax.imshow(img_data)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(output_path)
    # Explicitly close the figure to prevent memory warning
    plt.close(fig)

    print(f"Saved result to {output_path}")
    print("Transformation complete!")


def main():
    """Main function with argument parsing"""
    parser = argparse.ArgumentParser(
        description='Image Transformation using PlantCV - Extract 6 features from plant leaf images'
    )
    parser.add_argument('-i', '--image', type=str, help='Path to input image')
    parser.add_argument('-src', '--source', type=str, help='Path to source directory')
    parser.add_argument('-dst', '--destination', type=str, help='Path to destination directory')
    args = parser.parse_args()

    if args.image is None and args.source is None:
        parser.print_help()
        return

    # Transform all images in a directory
    if args.source is not None:
        if args.image is not None:
            print("Error: Cannot process both an image and a directory.")
            exit(1)
        if args.destination is None:
            print("Error: Destination directory must be specified when processing a directory.")
            exit(1)
        if not os.path.exists(args.source):
            print(f"Error: Source directory '{args.source}' not found!")
        for file in os.listdir(args.source):
            output_filename = f"transformed_{os.path.basename(file)}"
            output_path = os.path.join(args.destination, output_filename)
            process_single_image(os.path.join(args.source, file), output_path)

    # Process single image
    if args.image is not None:
        if args.destination is not None:
            print("Error: Destination directory must not be specified when processing a single image.")
            exit(1)
        if not os.path.exists(args.image):
            print(f"Error: Image file '{args.image}' not found!")
            exit(1)
        output_path = f"transformed_{os.path.basename(args.image)}"
        process_single_image(args.image, output_path)


if __name__ == '__main__':
    print("PlantCV ", pcv.__version__)
    main()

