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
Balances the dataset by applying various augmentation techniques (Flip, Rotate, Skew, Shear, Crop, Distortion). Generates 6 types of augmented images for each input.

**Usage:**
```bash
./augmentation.py <image_path>
```
*   When run, it creates copies of the image with different augmentations.
*   To balance the entire dataset, logic will be implemented to augment under-represented classes.

### 3. Image Transformation (`transformation.py`)
Applies feature extraction transformations to images, such as Gaussian blur, Masking, ROI extraction, etc.

**Usage:**
```bash
# For a single image (displays 6 transformations)
./transformation.py <image_path>

# For a directory (saves transformations)
./transformation.py -src <source_dir> -dst <destination_dir> [options]
```
Use `-h` for help.

### 4. Classification (`train.py` & `predict.py`)
Trains a model to classify leaf diseases and predicts the disease for new images.

**Training:**
```bash
./train.py <dataset_directory>
```
*   Generates a model and saves it (along with augmented data/learnings) to a zip file.

**Prediction:**
```bash
./predict.py <image_path>
```
*   Loads the trained model and predicts the disease class for the input image.
*   Displays the original image, transformed image, and the prediction.

## Requirements

*   Python 3.10+
*   See `pyproject.toml` or `setup.sh` for dependencies.

## Setup

Use the provided `setup.sh` script to manage the environment:

```bash
./setup.sh install  # Create venv and install dependencies
./setup.sh clean    # Clean venv and temporary files
```
