# Leaffliction Project - Technical Implementation Strategy

## Project Overview
The Leaffliction project is a computer vision application for plant disease classification using machine learning. The project consists of 4 main parts: data analysis, data augmentation, image transformation, and classification.

## Current Dataset Analysis
- **Apple diseases**: Black_rot (620), healthy (1640), rust (275), scab (629)
- **Grape diseases**: Black_rot (1178), Esca (1382), healthy (422), spot (1075)
- **Total images**: ~7,221 images
- **Dataset imbalance**: Significant variation in class sizes (275 to 1640 images per class)

## Technology Stack
- **Language**: Python 3.10+
- **Core Libraries**:
  - `plantcv`: Plant computer vision library (primary for image processing)
  - `opencv-python`: Computer vision operations
  - `numpy`: Numerical computations
  - `matplotlib`: Data visualization and plotting
  - `scikit-learn`: Machine learning algorithms
  - `scikit-image`: Additional image processing functions
- **Code Quality**: `flake8` for Python linting (norminette compliance)

## Implementation Strategy

### Part 1: Data Distribution Analysis (`Distribution.py`)

**Objective**: Analyze dataset distribution and create visualizations

**Implementation Plan**:
1. **Directory Structure Parsing**:
   - Recursively scan input directory for subdirectories
   - Extract plant type and disease categories from folder names
   - Count images in each category

2. **Data Analysis**:
   - Calculate distribution statistics per plant type
   - Identify class imbalances
   - Generate summary statistics

3. **Visualization**:
   - Create pie charts showing disease distribution per plant type
   - Generate bar charts comparing image counts across categories
   - Save charts with descriptive titles based on directory names

**Key Functions**:
- `scan_directory(path)`: Parse directory structure
- `count_images(path)`: Count images per category
- `generate_pie_chart(data, plant_type)`: Create pie chart
- `generate_bar_chart(data, plant_type)`: Create bar chart

### Part 2: Data Augmentation (`Augmentation.py`)

**Objective**: Balance dataset through image augmentation techniques

**Implementation Plan**:
1. **Augmentation Techniques** (6 required):
   - **Flip**: Horizontal/vertical flipping
   - **Rotate**: Random rotation (±15-45 degrees)
   - **Skew**: Perspective transformation
   - **Shear**: Shearing transformation
   - **Crop**: Random cropping and resizing
   - **Distortion**: Barrel/pincushion distortion

2. **Balancing Strategy**:
   - Calculate target count per class (use max class size)
   - Determine augmentation multiplier per class
   - Apply augmentations until balanced

3. **File Management**:
   - Save augmented images with suffix: `original_name_AugmentationType.JPG`
   - Create `augmented_directory` for evaluation
   - Maintain original directory structure

**Key Functions**:
- `apply_flip(image)`: Flip transformation
- `apply_rotation(image, angle)`: Rotation transformation
- `apply_skew(image)`: Skew transformation
- `apply_shear(image)`: Shear transformation
- `apply_crop(image)`: Crop transformation
- `apply_distortion(image)`: Distortion transformation
- `balance_dataset(input_dir, output_dir)`: Main balancing logic

### Part 3: Image Transformation (`Transformation.py`)

**Objective**: Extract leaf characteristics using plantCV

**Implementation Plan**:
1. **Core Transformations** (6+ required):
   - **Original**: Display original image
   - **Gaussian Blur**: Noise reduction using `cv2.GaussianBlur()`
   - **Mask**: Create binary mask using `plantcv.threshold.binary()`
   - **ROI Objects**: Extract regions of interest using `plantcv.roi.rectangle()`
   - **Analyze Object**: Object analysis using `plantcv.analyze.size()`
   - **Pseudolandmarks**: Generate landmarks using `plantcv.homology.pseudolandmarks()`
   - **Color Histogram**: Extract color features using `plantcv.analyze.color()`

2. **Command Line Interface**:
   - Single image mode: `./Transformation.py path/to/image.JPG`
   - Batch processing: `./Transformation.py -src input_dir -dst output_dir -mask`
   - Help option: `./Transformation.py -h`

3. **Output Management**:
   - Single image: Display transformations in matplotlib subplots
   - Batch mode: Save all transformations to destination directory

**Key Functions**:
- `load_image(path)`: Load and preprocess image
- `apply_gaussian_blur(image)`: Gaussian blur transformation
- `create_mask(image)`: Generate binary mask
- `extract_roi(image)`: Extract regions of interest
- `analyze_object(image, mask)`: Object analysis
- `generate_pseudolandmarks(image, mask)`: Create pseudolandmarks
- `generate_color_histogram(image, mask)`: Color analysis
- `process_single_image(path)`: Single image processing
- `process_batch(src_dir, dst_dir)`: Batch processing

### Part 4: Classification (`train.py` and `predict.py`)

**Objective**: Train ML model for disease classification with >90% validation accuracy

**Implementation Plan**:

#### Training (`train.py`):
1. **Data Pipeline**:
   - Load balanced dataset from augmented directory
   - Split data: 80% training, 20% validation (minimum 100 validation images)
   - Apply preprocessing (resize, normalize)

2. **Feature Extraction**:
   - Use plantCV for feature extraction (color histograms, shape features)
   - Extract statistical features (mean, std, texture features)
   - Combine multiple feature types

3. **Model Architecture**:
   - **Primary**: Random Forest or SVM (scikit-learn)
   - **Alternative**: Simple CNN using basic layers
   - Cross-validation for hyperparameter tuning

4. **Training Process**:
   - Stratified split to maintain class balance
   - Feature scaling/normalization
   - Model training with validation monitoring
   - Save model, scaler, and label encoder

5. **Output**:
   - Create ZIP file containing:
     - Trained model (`model.pkl`)
     - Feature scaler (`scaler.pkl`)
     - Label encoder (`label_encoder.pkl`)
     - Augmented dataset
     - Training metadata

#### Prediction (`predict.py`):
1. **Model Loading**:
   - Load trained model from ZIP file
   - Load preprocessing components

2. **Image Processing**:
   - Apply same preprocessing as training
   - Extract features using same pipeline

3. **Prediction**:
   - Display original and preprocessed image
   - Predict disease class with confidence score
   - Show prediction results

**Key Functions**:
- `load_dataset(path)`: Load and split dataset
- `extract_features(image)`: Feature extraction pipeline
- `train_model(X, y)`: Model training
- `save_model_zip(model, data, path)`: Save model and data
- `load_model_zip(path)`: Load trained model
- `predict_disease(image_path)`: Make prediction

### Project Structure
```
Leaffliction/
├── Distribution.py          # Part 1: Data analysis
├── Augmentation.py         # Part 2: Data augmentation
├── Transformation.py       # Part 3: Image transformations
├── train.py               # Part 4: Model training
├── predict.py             # Part 4: Prediction
├── pyproject.toml         # Project dependencies
├── signature.txt          # Dataset hash signature
├── README.md             # Project documentation
├── leaves/               # Original dataset
│   └── images/
├── augmented_directory/  # Balanced dataset (created)
└── model_output/        # Trained model ZIP (created)
```

### Development Workflow

1. **Environment Setup**:
   ```bash
   pip install -e .
   flake8 --install-hook git
   ```

2. **Implementation Order**:
   1. `Distribution.py` - Understand dataset characteristics
   2. `Augmentation.py` - Create balanced dataset
   3. `Transformation.py` - Implement feature extraction
   4. `train.py` - Develop and train model
   5. `predict.py` - Create prediction interface

3. **Testing Strategy**:
   - Unit tests for each transformation function
   - Integration tests for full pipeline
   - Validation accuracy testing (>90% requirement)
   - Small dataset testing for evaluation preparation

### Quality Assurance

1. **Code Quality**:
   - Follow PEP 8 standards
   - Use flake8 for linting
   - Add docstrings and type hints
   - Error handling for edge cases

2. **Performance Requirements**:
   - Handle large datasets efficiently
   - Memory-conscious image processing
   - Fast inference for real-time prediction

3. **Validation Requirements**:
   - Achieve >90% validation accuracy
   - Minimum 100 images in validation set
   - Prevent overfitting through proper validation
   - Cross-validation for robust results

### Deliverables

1. **Code Files**: All Python scripts with proper error handling
2. **Documentation**: README with usage instructions
3. **Model Package**: ZIP file with trained model and augmented dataset
4. **Signature File**: SHA1 hash of the model ZIP file
5. **Test Suite**: Scripts for validation and testing

### Risk Mitigation

1. **Dataset Issues**: Handle various image formats and corrupted files
2. **Memory Constraints**: Implement batch processing for large datasets
3. **Model Performance**: Multiple model architectures as backup
4. **Evaluation Preparation**: Pre-process small test datasets
5. **Dependency Management**: Clear installation instructions and version pinning

This strategy ensures compliance with all project requirements while maintaining code quality and achieving the required performance metrics.