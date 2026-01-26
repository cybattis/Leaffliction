# Leaffliction Implementation Plan

## overview
This plan outlines the steps to implement the Leaffliction project, which involves image analysis, augmentation, transformation, and classification of plant leaf diseases.

## Steps

### Part 1: Data Analysis
- [x] Create `distribution.py`.
- [x] Implement directory traversal to find images.
- [x] analyze class distribution (Healthy vs Diseases).
- [x] Generate Pie Charts and Bar Charts using Matplotlib/Seaborn.
- [x] Ensure charts are correctly labeled with directory source.

### Part 2: Data Augmentation
- [x] Create `augmentation.py`.
- [ ] Implement image loading.
- [ ] Implement 6 distinct augmentation techniques:
    - Flip
    - Rotate
    - Skew
    - Shear
    - Crop
    - Distortion
- [ ] Save augmented images with suffix (e.g., `_Flip.JPG`).
- [ ] Implement logic to process a directory and balance the dataset (augment under-represented classes to match the count of the most populated class).

### Part 3: Image Transformation
- [x] Create `transformation.py`.
- [ ] Implement feature extraction techniques (using `plantcv` or `opencv`):
    - Gaussian Blur
    - Masking
    - ROI Extraction
    - Object Analysis
    - Pseudolandmarks
    - Color Histogram
- [ ] Handle single image input: Display 6 transformations using Matplotlib.
- [ ] Handle directory input (`-src`, `-dst`): Apply transformations and save images to destination.
- [ ] Add command-line argument parsing (`argparse`).

### Part 4: Classification
- [x] Create `train.py`.
    - [ ] Load validation/training split.
    - [ ] Train a model (Logistic Regression, SVM, or simple MLP/CNN if allowed/feasible) on transformed/augmented data.
    - [ ] Save the model and learnings to a zip file.
- [x] Create `predict.py`.
    - [ ] Load the trained model.
    - [ ] Accept an image path.
    - [ ] Predict the disease.
    - [ ] Display Original, Transformed, and Prediction.
- [ ] Ensure validation accuracy > 90%
- [ ] Create `signature.txt` generation logic (SHA1 of contents).

## Notes
- Use `argparse` for all CLI handling.
- Ensure all code is modular and follows Python best practices.
- Use the virtual environment managed by `setup.sh`.
