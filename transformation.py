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
from plantcv.parallel import WorkflowInputs


def gaussian_blur_transform(img):
    """Apply Gaussian blur to reduce noise"""
    grayscale_img = pcv.rgb2gray(img)
    threshold_dark = pcv.threshold.binary(gray_img=grayscale_img, threshold=100, object_type='dark')
    gaussian_img = pcv.gaussian_blur(img=threshold_dark, ksize=(5, 5), sigma_x=0)
    return gaussian_img


def mask_transform(img):
    """Create a mask to isolate the disease (non-green) parts of the leaf"""
    # 1. Isolate the whole plant using 'b' channel (Yellow-Blue axis)
    # Plants (green/yellow/brown) are usually high in 'b' (Yellowish), background is usually low/neutral
    b = pcv.rgb2gray_lab(rgb_img=img, channel='b')
    mask_plant = pcv.threshold.binary(gray_img=b, threshold=100, object_type='light')

    # 2. Isolate healthy green parts using 'a' channel (Green-Magenta axis)
    # Healthy leaves are green (low 'a' values)
    a = pcv.rgb2gray_lab(rgb_img=img, channel='a')
    mask_green = pcv.threshold.binary(gray_img=a, threshold=105, object_type='dark')

    # 3. Combine: Disease = (Plant) AND (NOT Green)
    # Invert green mask to get everything that is NOT green
    mask_not_green = pcv.invert(gray_img=mask_green)

    # Intersection: Must be part of the plant AND not be green
    mask_disease = pcv.logical_and(bin_img1=mask_plant, bin_img2=mask_not_green)

    # Clean up mask
    mask_fill = pcv.fill(bin_img=mask_disease, size=200)

    # Apply mask to show only disease spots on black background
    masked = pcv.apply_mask(img=img, mask=mask_fill, mask_color='white')

    return masked


def roi_extraction(img):
    """Extract Region of Interest (ROI) using contour detection"""
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply threshold
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        return img

    # Get the largest contour
    largest_contour = max(contours, key=cv2.contourArea)

    # Get bounding box
    x, y, w, h = cv2.boundingRect(largest_contour)

    # Draw rectangle on copy
    roi_img = img.copy()
    cv2.rectangle(roi_img, (x, y), (x + w, y + h), (0, 255, 0), 3)

    return roi_img


def object_analysis(img):
    """Perform object analysis and draw contours"""
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply threshold
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw contours on copy
    obj_img = img.copy()
    cv2.drawContours(obj_img, contours, -1, (0, 255, 255), 2)

    # Add contour information text
    if len(contours) > 0:
        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)
        perimeter = cv2.arcLength(largest_contour, True)

        cv2.putText(obj_img, f'Area: {int(area)}', (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(obj_img, f'Perimeter: {int(perimeter)}', (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    return obj_img


def pseudolandmarks_transform(img):
    """Extract pseudolandmarks from the object"""
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply threshold
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    landmark_img = img.copy()

    if len(contours) > 0:
        largest_contour = max(contours, key=cv2.contourArea)

        # Get contour points
        contour_points = largest_contour.reshape(-1, 2)

        # Sample pseudolandmarks (every nth point)
        step = max(1, len(contour_points) // 20)
        landmarks = contour_points[::step]

        # Draw landmarks
        for i, point in enumerate(landmarks):
            cv2.circle(landmark_img, tuple(point), 5, (0, 0, 255), -1)

        # Draw lines connecting landmarks
        for i in range(len(landmarks) - 1):
            cv2.line(landmark_img, tuple(landmarks[i]), tuple(landmarks[i + 1]), 
                    (255, 0, 0), 1)

    return landmark_img


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


def process_single_image(image_path):
    """Process a single image and display all 6 transformations"""
    # Load image
    img, _, _ = pcv.readimage(filename=image_path)

    if img is None:
        print(f"Error: Could not load image from {image_path}")
        return

    print(f"Processing image: {image_path}")
    print(f"Image shape: {img.shape}")

    # Apply all transformations
    blur_img = gaussian_blur_transform(img)
    masked_img = mask_transform(img)
    # roi_img = roi_extraction(masked_img)
    # obj_img = object_analysis(masked_img)
    # landmark_img = pseudolandmarks_transform(masked_img)
    # hist_img = color_histogram_transform(img)

    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f'PlantCV Feature Extraction - {os.path.basename(image_path)}', 
                 fontsize=16, fontweight='bold')

    # Display images (convert BGR to RGB for matplotlib)
    transformations = [
        (cv2.cvtColor(img, cv2.COLOR_BGR2RGB), 'Original Image (BGR)'),
        (cv2.cvtColor(blur_img, cv2.COLOR_BGR2RGB), 'Gaussian Blur'),
        (cv2.cvtColor(masked_img, cv2.COLOR_BGR2RGB), 'Masking'),
        # (cv2.cvtColor(roi_img, cv2.COLOR_BGR2RGB), 'ROI Extraction'),
        # (cv2.cvtColor(obj_img, cv2.COLOR_BGR2RGB), 'Object Analysis'),
        # (cv2.cvtColor(landmark_img, cv2.COLOR_BGR2RGB), 'Pseudolandmarks'),
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
    parser.add_argument('-image', type=str, help='Path to input image')
    args = parser.parse_args()

    # Check if file exists
    if not os.path.exists(args.image):
        print(f"Error: Image file '{args.image}' not found!")
        return

    # Process the image
    process_single_image(args.image)


if __name__ == '__main__':
    print("PlantCV ", pcv.__version__)
    main()

