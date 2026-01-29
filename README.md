# Leaffliction

**Computer Vision Project - Image classification by disease recognition on leaves.**

## Overview

Leaffliction is a project focused on using computer vision techniques to identify and classify diseases in plant leaves. The project involves analyzing a dataset, performing data augmentation to balance the dataset, transforming images to extract features, and finally training a classification model.

## Programs

The project consists of four main parts, each with a specific program:

### 1. Analysis of the Data Set (`distribution.py`)
Analyzes the distribution of the dataset and generates pie charts and bar charts for each plant type.

**Usage:**
```bash
./distribution.py <directory_path>
```

### 2. Data Augmentation (`augmentation.py`)
Balances the dataset by applying various augmentation techniques (Flip, Rotate, Skew, Shear, Crop, Distortion).

**Usage:**
```bash
# Visualize augmentations for a single image
./augmentation.py -v <image_path>

# Generate augmented dataset for a specific plant type
./augmentation.py <apple|grape>
```
*   The script calculates the required number of images to balance classes and generates them.
*   Validation images (processed individually) are saved to `visualization/`.

### 3. Image Transformation (`transformation.py`)
Applies feature extraction using **PlantCV** to isolate the leaf and analyze its properties. It generates visualizations for **8 different processing steps** (including Masking, ROI extraction, Object Analysis, Pseudolandmarks) and creates specific color histograms.

**Usage:**
```bash
# Process a single image (saves to visualization/ folder)
./transformation.py -i <image_path>

# Process an entire directory
./transformation.py -src <source_dir> -dst <destination_dir> [options]
```
Use `-h` for help.

### 4. Classification (`train.py` & `predict.py`)
Trains a model to classify leaf diseases and predicts the disease for new images.

**Training:**
```bash
./train.py <apple|grape>
```
*   Running training automatically triggers the pipeline: Augmentation -> Transformation -> Training.
*   Generates a model and saves it (along with augmented data/learnings) to a zip file.

**Prediction:**
```bash
# Predict all images in a folder
./predict.py <folder_path> -m <model_name>

# Predict a single image (creates visualization)
./predict.py -i <image_path> -m <model_name>
```
*   Loads the trained model and predicts the disease class.
*   Single image prediction generates a side-by-side view with the binary mask.

## Requirements

*   Python 3.10+
*   See `pyproject.toml` or `setup.sh` for dependencies.

## Setup

Use the provided `setup.sh` script to manage the environment:

```bash
./setup.sh install  # Create venv and install dependencies
./setup.sh clean    # Clean venv and temporary files
```
