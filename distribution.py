#!/usr/bin/env python3
"""
Distribution.py - Data Distribution Analysis for Leaffliction Project

This module analyzes the dataset distribution and creates visualizations
including pie charts and bar charts for plant disease classification data.
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np


# Supported image extensions
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp'}


def scan_directory(path: str) -> Dict[str, Dict[str, int]]:
    """
    Parse directory structure and count images per category.

    Args:
        path: Path to the root directory containing class subdirectories.

    Returns:
        Dictionary with plant types as keys and disease counts as values.
        Format: {plant_type: {disease: count, ...}, ...}
    """
    distribution: Dict[str, Dict[str, int]] = {}
    root_path = Path(path)

    if not root_path.exists():
        raise FileNotFoundError(f"Directory not found: {path}")

    if not root_path.is_dir():
        raise NotADirectoryError(f"Path is not a directory: {path}")

    # Iterate through subdirectories
    for subdir in sorted(root_path.iterdir()):
        if not subdir.is_dir():
            continue

        # Parse folder name: expected format "PlantType_Disease"
        folder_name = subdir.name
        parts = folder_name.split('_', 1)

        if len(parts) == 2:
            plant_type, disease = parts
        else:
            # If no underscore, use folder name as both plant and disease
            plant_type = folder_name
            disease = folder_name

        # Count images in this subdirectory
        image_count = count_images(str(subdir))

        # Add to distribution
        if plant_type not in distribution:
            distribution[plant_type] = {}

        distribution[plant_type][disease] = image_count

    return distribution


def count_images(path: str) -> int:
    """
    Count the number of image files in a directory.

    Args:
        path: Path to the directory to scan.

    Returns:
        Number of image files found.
    """
    count = 0
    dir_path = Path(path)

    for file_path in dir_path.iterdir():
        if file_path.is_file():
            if file_path.suffix.lower() in IMAGE_EXTENSIONS:
                count += 1

    return count


def get_total_distribution(
    distribution: Dict[str, Dict[str, int]]
) -> Dict[str, int]:
    """
    Get total image count per class (plant_disease combination).

    Args:
        distribution: Distribution dictionary from scan_directory.

    Returns:
        Dictionary with class names as keys and counts as values.
    """
    total: Dict[str, int] = {}

    for plant_type, diseases in distribution.items():
        for disease, count in diseases.items():
            class_name = f"{plant_type}_{disease}"
            total[class_name] = count

    return total


def generate_combined_chart(
    data: Dict[str, int],
    title: str,
    output_path: str = None,
    show: bool = True,
    color_palette: str = 'Set2'
) -> None:
    """
    Generate a combined pie and bar chart for a single category.

    Args:
        data: Dictionary with category names and counts.
        title: Title for the chart.
        output_path: Optional path to save the chart.
        show: Whether to display the chart.
        color_palette: Matplotlib colormap name for colors.
    """
    if not data:
        print("No data to plot.")
        return

    labels = list(data.keys())
    sizes = list(data.values())
    total = sum(sizes)

    # Create color palette
    colors = [plt.colormaps[color_palette](i / len(labels))
              for i in range(len(labels))]

    # Create figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Pie chart (left)
    wedges, texts, autotexts = ax1.pie(
        sizes,
        labels=labels,
        autopct=lambda pct: f'{pct:.1f}%',
        colors=colors,
        startangle=90,
        wedgeprops={'edgecolor': 'black', 'linewidth': 1}
    )
    # Style the percentage text (white and bold)
    for autotext in autotexts:
        autotext.set_fontsize(9)
        autotext.set_color('white')
        autotext.set_fontweight('bold')
    for text in texts:
        text.set_fontsize(10)
    ax1.set_title('Distribution (%)', fontsize=12, fontweight='bold')
    ax1.axis('equal')

    # Bar chart (right)
    bars = ax2.bar(
        labels, sizes,
        color=colors,
        edgecolor='black',
        linewidth=0.5)
    for bar, count in zip(bars, sizes):
        height = bar.get_height()
        ax2.annotate(
            f'{count:,}',
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha='center',
            va='bottom',
            fontsize=9,
            fontweight='bold'
        )
    ax2.set_xlabel('Class', fontsize=11)
    ax2.set_ylabel('Number of Images', fontsize=11)
    ax2.set_title('Image Count', fontsize=12, fontweight='bold')
    ax2.tick_params(axis='x', rotation=45)
    for label in ax2.get_xticklabels():
        label.set_ha('right')
    ax2.yaxis.grid(True, linestyle='--', alpha=0.7)
    ax2.set_axisbelow(True)

    # Main title
    fig.suptitle(f'{title} (Total: {total:,} images)',
                 fontsize=14, fontweight='bold', y=1.02)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Chart saved to: {output_path}")

    if show:
        plt.show()
    else:
        plt.close()


def generate_overall_chart(
    distribution: Dict[str, Dict[str, int]],
    output_path: str = None,
    show: bool = True
) -> None:
    """
    Generate a combined chart showing all classes across all plant types.

    Args:
        distribution: Distribution dictionary from scan_directory.
        output_path: Optional path to save the chart.
        show: Whether to display the chart.
    """
    # Combine all data
    all_data: Dict[str, int] = {}
    for plant_type, diseases in distribution.items():
        for disease, count in diseases.items():
            class_name = f"{plant_type}_{disease}"
            all_data[class_name] = count

    if not all_data:
        print("No data to plot.")
        return

    labels = list(all_data.keys())
    sizes = list(all_data.values())
    total = sum(sizes)

    # Create better color palette based on plant type with high contrast
    plant_types = list(distribution.keys())

    # Use distinct colors for each plant type
    if len(plant_types) == 2:  # Apple and Grape
        apple_colors = [
            '#e74c3c',
            '#c0392b',
            '#a93226',
            '#922b21']
        grape_colors = [
            '#8e44ad',
            '#7d3c98',
            '#6c3483',
            '#5b2c6f']
        plant_color_schemes = {
            'Apple': apple_colors,
            'Grape': grape_colors
        }
    else:
        # Fallback for other plant types
        plant_color_schemes = {}
        base_colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6']
        for i, plant in enumerate(plant_types):
            base_color = base_colors[i % len(base_colors)]
            plant_color_schemes[plant] = [base_color]

    colors = []
    for label in labels:
        plant_type = label.split('_')[0]
        disease_idx = list(distribution[plant_type].keys()).index(
            label.split('_', 1)[1]
        )
        color_scheme = plant_color_schemes.get(plant_type, ['gray'])
        colors.append(color_scheme[disease_idx % len(color_scheme)])

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # Pie chart (left) with black borders
    wedges, texts, autotexts = ax1.pie(
        sizes,
        labels=labels,
        autopct=lambda pct: f'{pct:.1f}%',
        colors=colors,
        startangle=90,
        wedgeprops={'edgecolor': 'black', 'linewidth': 1.5}
    )
    # Style the percentage text (white and bold)
    for autotext in autotexts:
        autotext.set_fontsize(8)
        autotext.set_color('white')
        autotext.set_fontweight('bold')
    for text in texts:
        text.set_fontsize(9)
    ax1.set_title('Distribution (%)', fontsize=12, fontweight='bold')
    ax1.axis('equal')

    # Bar chart (right)
    bars = ax2.bar(labels,
                   sizes,
                   color=colors,
                   edgecolor='black',
                   linewidth=0.5)
    for bar, count in zip(bars, sizes):
        height = bar.get_height()
        ax2.annotate(
            f'{count:,}',
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha='center',
            va='bottom',
            fontsize=8,
            fontweight='bold'
        )
    ax2.set_xlabel('Class', fontsize=11)
    ax2.set_ylabel('Number of Images', fontsize=11)
    ax2.set_title('Image Count', fontsize=12, fontweight='bold')
    ax2.tick_params(axis='x', rotation=45)
    for label in ax2.get_xticklabels():
        label.set_ha('right')
    ax2.yaxis.grid(True, linestyle='--', alpha=0.7)
    ax2.set_axisbelow(True)

    # Add legend for plant types using the first color from each scheme
    legend_handles = []
    legend_labels = []
    for plant in plant_types:
        first_color = plant_color_schemes[plant][0]
        legend_handles.append(
            plt.Rectangle((0, 0), 1, 1, facecolor=first_color))
        legend_labels.append(plant)
    ax2.legend(legend_handles, legend_labels,
               title='Plant Type', loc='upper right')

    # Main title
    fig.suptitle(f'Overall Dataset Distribution (Total: {total:,} images)',
                 fontsize=14, fontweight='bold', y=1.02)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Chart saved to: {output_path}")

    if show:
        plt.show()
    else:
        plt.close()


def generate_plant_charts(
    distribution: Dict[str, Dict[str, int]],
    output_dir: str = None,
    show: bool = True
) -> None:
    """
    Generate combined pie+bar charts for each plant type.

    Args:
        distribution: Distribution dictionary from scan_directory.
        output_dir: Optional directory to save charts.
        show: Whether to display the charts.
    """
    for plant_type, diseases in distribution.items():
        title = f"{plant_type} Disease Distribution"

        output_path = None
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, f"{plant_type}.png")

        generate_combined_chart(diseases, title, output_path, show)


def print_summary(distribution: Dict[str, Dict[str, int]]) -> None:
    """
    Print a summary of the dataset distribution.

    Args:
        distribution: Distribution dictionary from scan_directory.
    """
    print("\n" + "=" * 60)
    print("DATASET DISTRIBUTION SUMMARY")
    print("=" * 60)

    total_images = 0
    all_counts = []

    for plant_type, diseases in sorted(distribution.items()):
        plant_total = sum(diseases.values())
        total_images += plant_total
        all_counts.extend(diseases.values())

        print(f"\n{plant_type}:")
        print("-" * 40)

        for disease, count in sorted(diseases.items(), key=lambda x: -x[1]):
            percentage = (count / plant_total) * 100
            print(f"  {disease:<20} {count:>6,} images ({percentage:>5.1f}%)")

        print(f"  {'TOTAL':<20} {plant_total:>6,} images")

    print("\n" + "=" * 60)
    print(f"GRAND TOTAL: {total_images:,} images")
    print(f"Number of classes: {len(all_counts)}")
    print(f"Min class size: {min(all_counts):,}")
    print(f"Max class size: {max(all_counts):,}")
    print(f"Average class size: {np.mean(all_counts):,.1f}")
    print(f"Class imbalance ratio: {max(all_counts) / min(all_counts):.2f}x")
    print("=" * 60 + "\n")


def parse_arguments() -> argparse.Namespace:
    """
    Parse command line arguments.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description="Analyze dataset distribution and generate charts.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python distribution.py leaves/images
  python distribution.py leaves/images --output charts/
  python distribution.py leaves/images --no-show --output charts/
        """
    )

    parser.add_argument(
        "directory",
        type=str,
        help="Path to the directory containing image class subdirectories."
    )

    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Directory to save generated charts."
    )

    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Don't display charts (only save if output is specified)."
    )

    return parser.parse_args()


def main() -> int:
    """
    Main entry point for the distribution analysis.

    Returns:
        Exit code (0 for success, non-zero for errors).
    """
    args = parse_arguments()

    try:
        # Scan directory and get distribution
        print(f"Scanning directory: {args.directory}")
        distribution = scan_directory(args.directory)

        if not distribution:
            print("No image classes found in the specified directory.")
            return 1

        # Print summary
        print_summary(distribution)

        # Determine whether to show charts
        show = not args.no_show

        # Generate overall combined chart (pie + bar for all classes)
        overall_output = None
        if args.output:
            os.makedirs(args.output, exist_ok=True)
            overall_output = os.path.join(args.output, "overall.png")

        generate_overall_chart(distribution, overall_output, show)

        # Generate per-plant combined charts (pie + bar for each plant)
        generate_plant_charts(distribution, args.output, show)

        print("Distribution analysis completed successfully.")
        return 0

    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except NotADirectoryError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
