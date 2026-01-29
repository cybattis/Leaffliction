#!/usr/bin/env python3

import os

import argparse
import json
import subprocess
import sys
import zipfile
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0:all, 1:info, 2:warning, 3:error
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN optimizations msg

# Constants
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001


def run_augmentation(
    input_dir: str,
    output_dir: str,
    plant_type: str
) -> bool:
    """
    Run augmentation.py to balance the dataset.

    Args:
        input_dir: Input directory containing original images.
        output_dir: Output directory for augmented images.
        plant_type: Plant type ('apple' or 'grape').

    Returns:
        True if augmentation succeeded, False otherwise.
    """
    print("\n" + "=" * 60)
    print("Running data augmentation to balance dataset...")
    print("=" * 60)

    # Build the augmentation command
    script_dir = os.path.dirname(os.path.abspath(__file__))
    augmentation_script = os.path.join(script_dir, "augmentation.py")

    if not os.path.exists(augmentation_script):
        print(f"Error: augmentation.py not found at {augmentation_script}")
        return False

    cmd = [
        sys.executable,
        augmentation_script,
        plant_type,  # positional argument
        "-i", input_dir,
        "-o", output_dir
    ]

    print(f"Command: {' '.join(cmd)}")

    try:
        subprocess.run(
            cmd,
            check=True,
            capture_output=False
        )
        print("\nAugmentation completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\nAugmentation failed with exit code {e.returncode}")
        return False
    except Exception as e:
        print(f"\nError running augmentation: {e}")
        return False


def run_transformation(
    input_dir: str,
    output_dir: str
) -> bool:
    """
    Run transformation.py to extract features from augmented images.

    Args:
        input_dir: Input directory containing augmented images.
        output_dir: Output directory for transformed images.

    Returns:
        True if transformation succeeded, False otherwise.
    """
    print("\n" + "=" * 60)
    print("Running image transformations to extract features...")
    print("=" * 60)

    # Build the transformation command
    script_dir = os.path.dirname(os.path.abspath(__file__))
    transformation_script = os.path.join(script_dir, "transformation.py")

    if not os.path.exists(transformation_script):
        print(f"Error: transformation.py not found at {transformation_script}")
        return False

    cmd = [
        sys.executable,
        transformation_script,
        "-src", input_dir,
        "-dst", output_dir,
        "-z"  # zip flag to skip individual histogram files
    ]

    print(f"Command: {' '.join(cmd)}")

    try:
        subprocess.run(
            cmd,
            check=True,
            capture_output=False
        )
        print("\nTransformation completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\nTransformation failed with exit code {e.returncode}")
        return False
    except Exception as e:
        print(f"\nError running transformation: {e}")
        return False


def create_training_zip(
    output_dir: str,
    augmented_dir: str,
    transformed_dir: str,
    model_name: str,
    plant_type: str
) -> str:
    """
    Create a zip file containing model, augmented images, and transformations.

    Args:
        output_dir: Directory containing model artifacts.
        augmented_dir: Directory containing augmented images.
        transformed_dir: Directory containing transformed images.
        model_name: Name of the model.
        plant_type: Plant type ('apple' or 'grape').

    Returns:
        Path to the created zip file.
    """
    print("\n" + "=" * 60)
    print("Creating training package (.zip)...")
    print("=" * 60)

    zip_filename = f"{plant_type}_model.zip"
    zip_path = os.path.join(os.path.dirname(output_dir), zip_filename)

    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # Add model files from output_dir
        print(f"Adding model artifacts from {output_dir}...")
        for root, dirs, files in os.walk(output_dir):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.join(
                    "model",
                    os.path.relpath(file_path, output_dir)
                )
                zipf.write(file_path, arcname)
                print(f"  + {arcname}")

        # Add augmented images
        print(f"\nAdding augmented images from {augmented_dir}...")
        image_count = 0
        for root, dirs, files in os.walk(augmented_dir):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    file_path = os.path.join(root, file)
                    arcname = os.path.join(
                        "augmented_images",
                        os.path.relpath(file_path, augmented_dir)
                    )
                    zipf.write(file_path, arcname)
                    image_count += 1

        print(f"  Added {image_count} images")

        # Add transformed images if directory exists
        if os.path.exists(transformed_dir):
            print(f"\nAdding transformed images from {transformed_dir}...")
            transform_count = 0
            for root, dirs, files in os.walk(transformed_dir):
                for file in files:
                    if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                        file_path = os.path.join(root, file)
                        arcname = os.path.join(
                            "transformed_images",
                            os.path.relpath(file_path, transformed_dir)
                        )
                        zipf.write(file_path, arcname)
                        transform_count += 1
            print(f"  Added {transform_count} transformed images")
        else:
            print(f"\nSkipping transformed images "
                  f"(directory not found: {transformed_dir})")

    zip_size_mb = os.path.getsize(zip_path) / (1024 * 1024)
    print(f"\nTraining package created: {zip_path}")
    print(f"   Size: {zip_size_mb:.1f} MB")

    return zip_path


def load_dataset(
    data_dir: str,
    validation_split: float = 0.2
) -> Tuple[tf.keras.preprocessing.image.DirectoryIterator,
           tf.keras.preprocessing.image.DirectoryIterator,
           List[str]]:
    """
    Load dataset from directory and split into training and validation sets.

    Args:
        data_dir: Path to the dataset directory with class subdirectories.
        validation_split: Fraction of data to use for validation.

    Returns:
        Tuple of (train_generator, val_generator, class_names).
    """
    print(f"\nLoading dataset from: {data_dir}")

    # Check if directory exists
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Dataset directory not found: {data_dir}")

    # Create data generators with preprocessing for EfficientNet
    # EfficientNet expects inputs in [0, 255] range, it has its own
    # preprocessing
    train_datagen = ImageDataGenerator(
        preprocessing_function=tf.keras.applications
        .efficientnet.preprocess_input,
        validation_split=validation_split,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        zoom_range=0.2
    )

    val_datagen = ImageDataGenerator(
        preprocessing_function=tf.keras.applications
        .efficientnet.preprocess_input,
        validation_split=validation_split
    )

    # Load training data
    train_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training',
        shuffle=True
    )

    # Load validation data
    val_generator = val_datagen.flow_from_directory(
        data_dir,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation',
        shuffle=False
    )

    # Get class names
    class_names = list(train_generator.class_indices.keys())

    print(f"Found {train_generator.samples} training images")
    print(f"Found {val_generator.samples} validation images")
    print(f"Classes: {class_names}")

    return train_generator, val_generator, class_names


def build_model(num_classes: int) -> keras.Model:
    """
    Build EfficientNetB0 model with custom classification head.

    Args:
        num_classes: Number of output classes.

    Returns:
        Compiled Keras model.
    """
    print(f"\nBuilding EfficientNetB0 model for {num_classes} classes...")

    # Load pre-trained EfficientNetB0
    base_model = EfficientNetB0(
        weights='imagenet',
        include_top=False,
        input_shape=(*IMAGE_SIZE, 3)
    )

    # Freeze base model layers initially
    base_model.trainable = False

    # Build model architecture
    inputs = keras.Input(shape=(*IMAGE_SIZE, 3))
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = keras.Model(inputs, outputs)

    # Compile model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    model.summary()

    return model


def create_callbacks(
    output_dir: str,
    model_name: str
) -> List[keras.callbacks.Callback]:
    """
    Create training callbacks.

    Args:
        output_dir: Directory to save model checkpoints.
        model_name: Name for the model files.

    Returns:
        List of Keras callbacks.
    """
    os.makedirs(output_dir, exist_ok=True)

    callbacks = [
        # Save best model based on validation accuracy
        ModelCheckpoint(
            filepath=os.path.join(output_dir, f"{model_name}_best.keras"),
            monitor='val_accuracy',
            mode='max',
            save_best_only=True,
            verbose=1
        ),
        # Early stopping to prevent overfitting
        EarlyStopping(
            monitor='val_accuracy',
            mode='max',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        # Reduce learning rate on plateau
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=5,
            min_lr=1e-7,
            verbose=1
        )
    ]

    return callbacks


def train_model(
    model: keras.Model,
    train_generator: tf.keras.preprocessing.image.DirectoryIterator,
    val_generator: tf.keras.preprocessing.image.DirectoryIterator,
    callbacks: List[keras.callbacks.Callback],
    epochs: int = EPOCHS
) -> keras.callbacks.History:
    """
    Train the model.

    Args:
        model: Keras model to train.
        train_generator: Training data generator.
        val_generator: Validation data generator.
        callbacks: List of callbacks.
        epochs: Number of epochs to train.

    Returns:
        Training history.
    """
    print(f"\nStarting training for {epochs} epochs...")

    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1
    )

    return history


def plot_training_history(
    history: keras.callbacks.History,
    output_dir: str,
    model_name: str
) -> None:
    """
    Plot and save training history curves.

    Args:
        history: Training history object.
        output_dir: Directory to save plots.
        model_name: Name for the plot files.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Accuracy plot
    axes[0].plot(history.history['accuracy'], label='Training Accuracy')
    axes[0].plot(history.history['val_accuracy'], label='Validation Accuracy')
    axes[0].set_title('Model Accuracy')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True)

    # Loss plot
    axes[1].plot(history.history['loss'], label='Training Loss')
    axes[1].plot(history.history['val_loss'], label='Validation Loss')
    axes[1].set_title('Model Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, f"{model_name}_training_history.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Training history plot saved to: {plot_path}")


def evaluate_model(
    model: keras.Model,
    val_generator: tf.keras.preprocessing.image.DirectoryIterator,
    class_names: List[str],
    output_dir: str,
    model_name: str
) -> Dict:
    """
    Evaluate model and generate confusion matrix.

    Args:
        model: Trained Keras model.
        val_generator: Validation data generator.
        class_names: List of class names.
        output_dir: Directory to save evaluation results.
        model_name: Name for the output files.

    Returns:
        Dictionary with evaluation metrics.
    """
    print("\nEvaluating model on validation set...")

    # Get predictions
    val_generator.reset()
    predictions = model.predict(val_generator, verbose=1)
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = val_generator.classes

    # Calculate metrics
    loss, accuracy = model.evaluate(val_generator, verbose=0)
    print(f"\nValidation Loss: {loss:.4f}")
    print(f"Validation Accuracy: {accuracy:.4f}")

    # Classification report
    report = classification_report(
        true_classes,
        predicted_classes,
        target_names=class_names,
        output_dict=True
    )
    print("\nClassification Report:")
    print(classification_report(
        true_classes,
        predicted_classes,
        target_names=class_names
    ))

    # Confusion matrix
    cm = confusion_matrix(true_classes, predicted_classes)
    fig, ax = plt.subplots(figsize=(10, 8))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=class_names
    )
    disp.plot(ax=ax, cmap='Blues', xticks_rotation=45)
    plt.title(f'Confusion Matrix - {model_name}')
    plt.tight_layout()

    cm_path = os.path.join(output_dir, f"{model_name}_confusion_matrix.png")
    plt.savefig(cm_path, dpi=150)
    plt.close()
    print(f"Confusion matrix saved to: {cm_path}")

    return {
        'accuracy': float(accuracy),
        'loss': float(loss),
        'classification_report': report
    }


def save_model_artifacts(
    model: keras.Model,
    class_names: List[str],
    history: keras.callbacks.History,
    metrics: Dict,
    output_dir: str,
    model_name: str
) -> None:
    """
    Save model and all training artifacts.

    Args:
        model: Trained Keras model.
        class_names: List of class names.
        history: Training history.
        metrics: Evaluation metrics.
        output_dir: Directory to save artifacts.
        model_name: Name for the model files.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Save final model
    model_path = os.path.join(output_dir, f"{model_name}_final.keras")
    model.save(model_path)
    print(f"Model saved to: {model_path}")

    # Save class labels
    labels_path = os.path.join(output_dir, f"{model_name}_labels.json")
    with open(labels_path, 'w') as f:
        json.dump({
            'class_names': class_names,
            'class_indices': {name: i for i, name in enumerate(class_names)}
        }, f, indent=2)
    print(f"Class labels saved to: {labels_path}")

    # Save training history
    history_path = os.path.join(output_dir, f"{model_name}_history.json")
    history_dict = {
        key: [float(v) for v in values]
        for key, values in history.history.items()
    }
    with open(history_path, 'w') as f:
        json.dump(history_dict, f, indent=2)
    print(f"Training history saved to: {history_path}")

    # Save metrics
    metrics_path = os.path.join(output_dir, f"{model_name}_metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to: {metrics_path}")

    # Save training metadata
    metadata = {
        'model_name': model_name,
        'timestamp': datetime.now().isoformat(),
        'image_size': IMAGE_SIZE,
        'batch_size': BATCH_SIZE,
        'epochs_trained': len(history.history['accuracy']),
        'final_accuracy': metrics['accuracy'],
        'final_loss': metrics['loss'],
        'num_classes': len(class_names),
        'class_names': class_names
    }
    metadata_path = os.path.join(output_dir, f"{model_name}_metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadata saved to: {metadata_path}")


def parse_arguments() -> argparse.Namespace:
    """
    Parse command line arguments.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description="Train EfficientNetB0 model for plant disease "
                    "classification.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train apple model (runs augmentation -> transformation -> training)
  python train.py apple

  # Train grape model with custom epochs
  python train.py grape -e 100 -b 64

  # Use custom source and output directories
  python train.py apple -s leaves/images -o models/

Pipeline:
  1. Runs augmentation.py to balance the dataset
  2. Runs transformation.py to extract features from augmented images
  3. Trains the model on augmented images
  4. Creates .zip with model + augmented + transformed images
        """
    )

    parser.add_argument(
        "plant_type",
        type=str,
        choices=["apple", "grape"],
        help="Plant type to train on (apple or grape)"
    )

    parser.add_argument(
        "-s", "--source-dir",
        type=str,
        default="leaves/images",
        help="Source directory with original images (default: leaves/images)"
    )

    parser.add_argument(
        "-a", "--augmented-dir",
        type=str,
        default=None,
        help="Directory for augmented images "
             "(default: augmented_{plant_type}/)"
    )

    parser.add_argument(
        "-o", "--output-dir",
        type=str,
        default="models",
        help="Output directory for model artifacts (default: models/)"
    )

    parser.add_argument(
        "-e", "--epochs",
        type=int,
        default=EPOCHS,
        help=f"Number of training epochs (default: {EPOCHS})"
    )

    parser.add_argument(
        "-b", "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help=f"Batch size for training (default: {BATCH_SIZE})"
    )

    return parser.parse_args()


def main() -> int:
    """
    Main entry point for the training script.

    Returns:
        Exit code (0 for success, non-zero for errors).
    """
    args = parse_arguments()

    # Update global constants if provided
    global BATCH_SIZE, EPOCHS
    BATCH_SIZE = args.batch_size
    EPOCHS = args.epochs

    # Set default augmented directory
    if args.augmented_dir is None:
        args.augmented_dir = f"augmented_{args.plant_type}/"

    # Set transformed directory
    transformed_dir = f"transformed_{args.plant_type}/"

    # Create model name
    model_name = f"{args.plant_type}_disease_model"
    output_dir = str(os.path.join(args.output_dir, args.plant_type))

    print("=" * 60)
    print("Leaffliction - Plant Disease Classification Training")
    print("=" * 60)
    print(f"Plant type: {args.plant_type}")
    print(f"Source directory: {args.source_dir}")
    print(f"Augmented directory: {args.augmented_dir}")
    print(f"Transformed directory: {transformed_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print("=" * 60)

    try:
        # Step 1: Run augmentation to balance dataset (skip if exists)
        if (os.path.exists(args.augmented_dir) and
                os.listdir(args.augmented_dir)):
            print("\n" + "=" * 60)
            print("Augmented dataset already exists, skipping augmentation...")
            print("=" * 60)
            print(f"Using existing: {args.augmented_dir}")
        else:
            if not run_augmentation(
                args.source_dir,
                args.augmented_dir,
                args.plant_type
            ):
                print("Failed to run augmentation.", file=sys.stderr)
                return 1

        # Step 2: Run transformation to extract features (skip if exists)
        if os.path.exists(transformed_dir) and os.listdir(transformed_dir):
            print("\n" + "=" * 60)
            print("Transformed dataset exists, skipping transformation...")
            print("=" * 60)
            print(f"Using existing: {transformed_dir}")
        else:
            if not run_transformation(
                args.augmented_dir,
                transformed_dir
            ):
                print("Failed to run transformation.", file=sys.stderr)
                return 1

        # Check for GPU
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"\nGPU(s) available: {len(gpus)}")
            for gpu in gpus:
                print(f"  - {gpu}")
        else:
            print("\nNo GPU found. Training will use CPU.")

        # Step 3: Load dataset (use augmented for training, not transformed)
        train_gen, val_gen, class_names = load_dataset(
            args.augmented_dir,
            validation_split=0.2
        )

        # Validate minimum validation samples
        if val_gen.samples < 100:
            print(f"\nWarning: Only {val_gen.samples} validation samples. "
                  f"Recommended minimum is 100.")

        # Step 4: Build model
        model = build_model(num_classes=len(class_names))

        # Create callbacks
        callbacks = create_callbacks(output_dir, model_name)

        # Step 5: Train model
        print("\n" + "=" * 60)
        print("Training with frozen EfficientNetB0 base")
        print("=" * 60)
        history = train_model(
            model, train_gen, val_gen, callbacks, epochs=args.epochs
        )

        # Step 6: Plot training history
        plot_training_history(history, output_dir, model_name)

        # Step 7: Evaluate model
        metrics = evaluate_model(
            model, val_gen, class_names, output_dir, model_name
        )

        # Step 8: Save all artifacts
        save_model_artifacts(
            model, class_names, history, metrics, output_dir, model_name
        )

        # Step 9: Create zip package with model, augmented, and transformed
        zip_path = create_training_zip(
            output_dir,
            args.augmented_dir,
            transformed_dir,
            model_name,
            args.plant_type
        )

        print("\n" + "=" * 60)
        print("Training Complete!")
        print("=" * 60)
        print(f"Final Validation Accuracy: {metrics['accuracy']:.4f}")
        print(f"Model saved to: {output_dir}")
        print(f"Training package: {zip_path}")
        print("=" * 60)

        # Check if target accuracy achieved
        if metrics['accuracy'] >= 0.90:
            print("Target accuracy (>90%) achieved!")
        else:
            print(f" Accuracy {metrics['accuracy']:.1%} below 90% target.")

        return 0

    except FileNotFoundError as e:
        print(f"\nError: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"\nUnexpected error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
