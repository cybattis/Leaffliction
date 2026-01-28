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


def plant_mask(img, lower_green=None, upper_green=None):
    """Create a mask to isolate the plant from the background"""
    # Default HSV ranges for plant green vegetation
    if lower_green is None:
        lower_green = (15, 40, 40) # H=35° (yellow-green), S=40 (some color), V=40 (not too dark)
    if upper_green is None:
        upper_green = (85, 255, 255) # H=85° (cyan-green), S=255 (vivid), V=255 (bright)


    blurred_img = pcv.gaussian_blur(img=img, ksize=(15, 15), sigma_x=0)
    bgr = cv2.cvtColor(blurred_img, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask, masked_img = pcv.threshold.custom_range(img=blurred_img,
                                                  lower_thresh=lower_green,
                                                  upper_thresh=upper_green,
                                                  channel='HSV')

    return hsv, mask, masked_img, blurred_img


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

    analysis_image = pcv.analyze.size(img=img, labeled_mask=kept_mask)
    return analysis_image


def roi_contour_extraction(original_img, mask):
    """Extract and visualize ROI"""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    roi_contours = original_img.copy()
    cv2.drawContours(roi_contours, contours, -1, (255, 0, 0), 3)

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


def color_histogram_transform(img, mask):
    """
    Analyzes color and returns the histogram plot as an image array
    Includes RGB, HSV, and LAB color spaces using plantcv.visualize.histogram
    """
    # 1. RGB Histogram (Original Image)
    # pcv.visualize.histogram plots pixel counts for channels
    hist_rgb = pcv.visualize.histogram(img=img, mask=mask,
                                       title="RGB Histogram")
    filename_rgb = "temp_hist_rgb.png"
    hist_rgb.save(filename_rgb)

    # 2. HSV Histogram (Hue, Saturation, Value)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hist_hsv = pcv.visualize.histogram(img=hsv, mask=mask,
                                       title="HSV Histogram")
    filename_hsv = "temp_hist_hsv.png"
    hist_hsv.save(filename_hsv)

    # 3. LAB Histogram (Lightness, Green-Magenta, Blue-Yellow)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
    hist_lab = pcv.visualize.histogram(img=lab, mask=mask,
                                       title="LAB Histogram")
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
        if source_img is None:
            return target_img
        _, w_t = target_img.shape[:2]
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


def process_single_image(settings, image_path, output_path):
    """Process a single image and display all 6 transformations"""
    # Preprocessing - Mask background
    original_img, _, _ = pcv.readimage(filename=image_path)
    if original_img is None:
        print(f"Error: Could not load image from {image_path}")
        return

    print(f"Processing image: {image_path}")
    print(f"Image shape: {original_img.shape}")

    hsv, mask, masked_img, blurred_img = plant_mask(original_img)
    only_plant = pcv.apply_mask(img=original_img, mask=mask, mask_color='black')
    diseases_mask = mask_transform(only_plant)

    blur_img = gaussian_blur_transform(only_plant)
    obj_img = object_analyse(original_img, diseases_mask)
    roi_contours, roi_analyse = roi_contour_extraction(original_img, mask)
    landmark_img = pseudolandmarks_transform(original_img, mask)
    hist_img = color_histogram_transform(original_img, mask)

    # Display images (convert BGR to RGB for matplotlib)
    if settings.debug:
        original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
        blurred_img = cv2.cvtColor(blurred_img, cv2.COLOR_BGR2RGB)
        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
        masked_img = cv2.cvtColor(masked_img, cv2.COLOR_BGR2RGB)
        only_plant = cv2.cvtColor(only_plant, cv2.COLOR_BGR2RGB)

        # Visualize the HSV channels and resulting mask
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
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
        axes[1, 0].imshow(hsv[:, :, 2], cmap='gray')
        axes[1, 0].set_title('Value Channel (Brightness)')
        axes[1, 0].axis('off')

        axes[1, 1].imshow(mask, cmap='gray')
        axes[1, 1].set_title('HSV Green Mask')
        axes[1, 1].axis('off')

        # Overlay mask on original
        overlay = original_img.copy()
        overlay[mask == 0] = overlay[mask == 0] // 2  # Darken background
        axes[1, 2].imshow(overlay)
        axes[1, 2].set_title('Mask Overlay')
        axes[1, 2].axis('off')

        plt.tight_layout()
        plt.savefig(output_path)
        plt.close(fig)
    else:
        transformations = [
            (cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB), 'Original Image'),
            (cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB), 'Plant Mask'),
            (cv2.cvtColor(only_plant, cv2.COLOR_BGR2RGB), 'Preprocess image'),
            (cv2.cvtColor(blur_img, cv2.COLOR_BGR2RGB), 'Gaussian Blur'),
            (cv2.cvtColor(roi_contours, cv2.COLOR_BGR2RGB), 'ROI Contour Extraction'),
            (cv2.cvtColor(roi_analyse, cv2.COLOR_BGR2RGB), 'ROI Analysis'),
            (cv2.cvtColor(obj_img, cv2.COLOR_BGR2RGB), 'Object Analysis'),
            (cv2.cvtColor(landmark_img, cv2.COLOR_BGR2RGB), 'Pseudolandmarks'),
            (cv2.cvtColor(hist_img, cv2.COLOR_BGR2RGB), 'Color Histogram')
        ]

        # Create figure with subplots
        fig, axes = plt.subplots(3, 3, figsize=(20, 15))
        fig.suptitle(f'PlantCV Feature Extraction - '
                     f'{os.path.basename(image_path)}',
                     fontsize=16, fontweight='bold')

        for ax, (img_data, title) in zip(axes.flat, transformations):
            ax.imshow(img_data)
            ax.set_title(title, fontsize=12, fontweight='bold')
            ax.axis('off')

        plt.tight_layout()
        plt.savefig(output_path)
        plt.close(fig)

    print(f"Saved result to {output_path}")
    print("Transformation complete!")


def main():
    parser = argparse.ArgumentParser(
        description='Image Transformation using PlantCV - '
                    'Extract 6 features from plant leaf images')
    parser.add_argument('-i', '--image', type=str,
                        help='Path to input image')
    parser.add_argument('-src', '--source', type=str,
                        help='Path to source directory')
    parser.add_argument('-dst', '--destination', type=str,
                        help='Path to destination directory')
    parser.add_argument('-d', '--debug', action='store_true')
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
            print("Error: Destination directory must be "
                  "specified when processing a directory.")
            exit(1)
        if not os.path.exists(args.source):
            print(f"Error: Source directory '{args.source}' not found!")
        for file in os.listdir(args.source):
            output_filename = f"transformed_{os.path.basename(file)}"
            output_path = os.path.join(args.destination, output_filename)
            process_single_image(args, os.path.join(args.source, file), output_path)

    # Process single image
    if args.image is not None:
        if args.destination is not None:
            print("Error: Destination directory must not "
                  "be specified when processing a single image.")
            exit(1)
        if not os.path.exists(args.image):
            print(f"Error: Image file '{args.image}' not found!")
            exit(1)
        output_path = f"transformed_{os.path.basename(args.image)}"
        process_single_image(args, args.image, output_path)


if __name__ == '__main__':
    print("PlantCV ", pcv.__version__)
    main()
