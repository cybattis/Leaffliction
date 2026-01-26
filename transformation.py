#!/usr/bin/env python3
"""
Image Transformation using PlantCV
Extracts 6 features from plant leaf images
"""
import os
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cv2

# Import libraries
from plantcv import plantcv as pcv


def gaussian_blur_transform(img):
    """Apply Gaussian blur to reduce noise"""
    grayscale_img = pcv.rgb2gray(img)
    threshold_dark = pcv.threshold.binary(gray_img=grayscale_img, threshold=100, object_type='dark')
    gaussian_img = pcv.gaussian_blur(img=threshold_dark, ksize=(5, 5), sigma_x=0)
    return gaussian_img


def mask_transform(img):
    """Create a mask to isolate the disease (non-green) parts of the leaf"""
    # 1. Isolate the whole plant using 'b' channel (Yellow-Blue axis)
    b = pcv.rgb2gray_lab(rgb_img=img, channel='b')
    mask_plant = pcv.threshold.binary(gray_img=b, threshold=100, object_type='light')

    # 2. Isolate healthy green parts using 'a' channel (Green-Magenta axis)
    a = pcv.rgb2gray_lab(rgb_img=img, channel='a')
    mask_green = pcv.threshold.binary(gray_img=a, threshold=105, object_type='dark')

    # 3. Combine: Disease = (Plant) AND (NOT Green)
    mask_not_green = pcv.invert(gray_img=mask_green)

    # Intersection: Must be part of the plant AND not be green
    mask_disease = pcv.logical_and(bin_img1=mask_plant, bin_img2=mask_not_green)

    # Clean up mask
    mask_fill = pcv.fill(bin_img=mask_disease, size=100)

    # Apply mask to show only disease spots on black background
    masked = pcv.apply_mask(img=img, mask=mask_fill, mask_color='white')

    return masked


def object_analyse(img, original_img):
    """Calcule le ROI et retourne une image avec le rectangle dessiné pour visualisation"""
    # pcv.roi.rectangle a besoin d'une image de référence (niveaux de gris)
    b_img = pcv.rgb2gray_lab(rgb_img=img, channel='b')
    thresh_mask = pcv.threshold.binary(gray_img=b_img, threshold=140, object_type='light')
    fill_mask = pcv.fill(bin_img=thresh_mask, size=1000)
    roi = pcv.roi.rectangle(img=fill_mask, x=0, y=0, h=256, w=256)
    kept_mask = pcv.roi.quick_filter(mask=fill_mask, roi=roi)
    analysis_image = pcv.analyze.size(img=original_img, labeled_mask=kept_mask)

    return analysis_image


def roiroiroi(img):
    """Perform object analysis and draw contours"""



def pseudolandmarks_transform(img, mask, original_img):
    """Extract pseudolandmarks from the object"""
    # Identify a set of land mark points
    # Results in set of point values that may indicate tip points
    gray_mask = pcv.rgb2gray(mask)
    binary_mask = pcv.threshold.binary(gray_img=gray_mask, threshold=130, object_type='dark')

    left, right, center_h = pcv.homology.x_axis_pseudolandmarks(img=img, mask=binary_mask)

    # 3. Create a copy of the image to draw on
    landmarks_img = original_img.copy()

    # 4. Helper to draw specific points
    def draw_points(points, color):
        if points is not None:
            for pt in points:
                # Points are returned as [[x, y]] lists
                x, y = pt[0]
                cv2.circle(landmarks_img, (int(x), int(y)), 5, color, -1)

    # 5. Draw landmarks (BGR Colors)
    # Left = Blue, Right = Red, Center = Green
    draw_points(left, (255, 0, 0))
    draw_points(right, (0, 0, 255))
    draw_points(center_h, (0, 255, 0))

    return landmarks_img


def color_histogram_transform(img):
    """Generate color histogram visualization"""
    # Create a figure for the histogram
    fig, ax = plt.subplots(figsize=(6, 4), dpi=100)

    # Calculate histograms for each channel
    colors = ('b', 'g', 'r')
    channel_names = ('Blue', 'Green', 'Red')

    for i, (color, name) in enumerate(zip(colors, channel_names)):
        hist = cv2.calcHist([img], [i], None, [256], [0, 256])
        ax.plot(hist, color=color, label=name)

    ax.set_xlim([0, 256])
    ax.set_xlabel('Pixel Value')
    ax.set_ylabel('Frequency')
    ax.set_title('Color Histogram')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Convert plot to image using a more compatible method
    fig.tight_layout()
    fig.canvas.draw()

    # Convert canvas to numpy array
    data = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (4,))
    data = data[:, :, :3]

    plt.close(fig)

    # Convert RGB to BGR for consistency with OpenCV
    hist_img = cv2.cvtColor(data, cv2.COLOR_RGB2BGR)

    return hist_img


def exclude_shadows(img, mask_color: str = "white"):
    v_channel = pcv.rgb2gray_hsv(rgb_img=img, channel='v')
    s_thresh = pcv.threshold.binary(gray_img=v_channel, threshold=25, object_type='light')
    s_clean = pcv.fill(bin_img=s_thresh, size=200)
    no_shadows_img = pcv.apply_mask(img=img, mask=s_clean, mask_color=mask_color.lower())
    return no_shadows_img


def exclude_background(img, mask_color: str = "white"):
    b_channel = pcv.rgb2gray_hsv(rgb_img=img, channel='h')
    b_thresh = pcv.threshold.binary(gray_img=b_channel, threshold=125, object_type='dark')
    b_clean = pcv.fill(bin_img=b_thresh, size=200)
    no_bg_img = pcv.apply_mask(img=img, mask=b_clean, mask_color=mask_color.upper())
    return no_bg_img


def process_single_image(image_path):
    """Process a single image and display all 6 transformations"""
    # Preprocessing - Mask background
    img, _, _ = pcv.readimage(filename=image_path)
    if img is None:
        print(f"Error: Could not load image from {image_path}")
        return

    print(f"Processing image: {image_path}")
    print(f"Image shape: {img.shape}")

    original_img = img.copy()

    # Exclude shadows and background
    img = exclude_shadows(img)
    img = exclude_background(img)

    # Apply all transformations
    blur_img = gaussian_blur_transform(img)
    masked_img = mask_transform(img)
    obj_img = object_analyse(img, original_img)
    # obj_img = roiroiroi(img)
    landmark_img = pseudolandmarks_transform(img, masked_img, original_img)
    # hist_img = color_histogram_transform(img)

    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f'PlantCV Feature Extraction - {os.path.basename(image_path)}', 
                 fontsize=16, fontweight='bold')

    # Display images (convert BGR to RGB for matplotlib)
    transformations = [
        (cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB), 'Original Image (BGR)'),
        (cv2.cvtColor(img, cv2.COLOR_BGR2RGB), 'Preprocess image'),
        (cv2.cvtColor(blur_img, cv2.COLOR_BGR2RGB), 'Gaussian Blur'),
        (cv2.cvtColor(masked_img, cv2.COLOR_BGR2RGB), 'Masking'),
        (cv2.cvtColor(obj_img, cv2.COLOR_BGR2RGB), 'Object Analysis'),
        # (cv2.cvtColor(obj_img, cv2.COLOR_BGR2RGB), 'Object Analysis'),
        (cv2.cvtColor(landmark_img, cv2.COLOR_BGR2RGB), 'Pseudolandmarks'),
        # (cv2.cvtColor(hist_img, cv2.COLOR_BGR2RGB), 'Color Histogram')
    ]

    for ax, (img_data, title) in zip(axes.flat, transformations):
        ax.imshow(img_data)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.axis('off')

    plt.tight_layout()

    output_path = f"transformed_{os.path.basename(image_path)}"
    plt.savefig(output_path)
    print(f"Saved result to {output_path}")
    # plt.show()

    print("Transformation complete!")


def main():
    """Main function with argument parsing"""
    parser = argparse.ArgumentParser(
        description='Image Transformation using PlantCV - Extract 6 features from plant leaf images'
    )
    parser.add_argument('-i', '--image', type=str, help='Path to input image')
    args = parser.parse_args()

    if args.image is None:
        parser.print_help()
        return

    # Check if file exists
    if not os.path.exists(args.image):
        print(f"Error: Image file '{args.image}' not found!")
        return

    # Process the image
    process_single_image(image_path=args.image)


if __name__ == '__main__':
    print("PlantCV ", pcv.__version__)
    main()

