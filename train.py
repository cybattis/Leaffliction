#!/usr/bin/env python3

import os

import argparse
import json
import sys
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

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0=all, 1=info, 2=warning, 3=error
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN optimizations msg

# Constants
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001


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
  python train.py apple
  python train.py grape
  python train.py apple -i custom_data/ -o models/
  python train.py grape -e 100 -b 64
        """
    )

    parser.add_argument(
        "plant_type",
        type=str,
        choices=["apple", "grape"],
        help="Plant type to train on (apple or grape)"
    )

    parser.add_argument(
        "-i", "--input-dir",
        type=str,
        default=None,
        help="Input directory with balanced dataset "
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

    # Set default input directory
    if args.input_dir is None:
        args.input_dir = f"augmented_{args.plant_type}/"

    # Create model name
    model_name = f"{args.plant_type}_disease_model"
    output_dir = os.path.join(args.output_dir, args.plant_type)

    print("=" * 60)
    print("Leaffliction - Plant Disease Classification Training")
    print("=" * 60)
    print(f"Plant type: {args.plant_type}")
    print(f"Input directory: {args.input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print("=" * 60)

    try:
        # Check for GPU
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"\nGPU(s) available: {len(gpus)}")
            for gpu in gpus:
                print(f"  - {gpu}")
        else:
            print("\nNo GPU found. Training will use CPU.")

        # Load dataset
        train_gen, val_gen, class_names = load_dataset(
            args.input_dir,
            validation_split=0.2
        )

        # Validate minimum validation samples
        if val_gen.samples < 100:
            print(f"\nWarning: Only {val_gen.samples} validation samples. "
                  f"Recommended minimum is 100.")

        # Build model
        model = build_model(num_classes=len(class_names))

        # Create callbacks
        callbacks = create_callbacks(output_dir, model_name)

        # Train model
        print("\n" + "=" * 60)
        print("Training with frozen EfficientNetB0 base")
        print("=" * 60)
        history = train_model(
            model, train_gen, val_gen, callbacks, epochs=args.epochs
        )

        # Plot training history
        plot_training_history(history, output_dir, model_name)

        # Evaluate model
        metrics = evaluate_model(
            model, val_gen, class_names, output_dir, model_name
        )

        # Save all artifacts
        save_model_artifacts(
            model, class_names, history, metrics, output_dir, model_name
        )

        print("\n" + "=" * 60)
        print("Training Complete!")
        print("=" * 60)
        print(f"Final Validation Accuracy: {metrics['accuracy']:.4f}")
        print(f"Model saved to: {output_dir}")
        print("=" * 60)

        # Check if target accuracy achieved
        if metrics['accuracy'] >= 0.90:
            print("✅ Target accuracy (>90%) achieved!")
        else:
            print(f"⚠️  Accuracy {metrics['accuracy']:.1%} below 90% target. "
                  "Consider fine-tuning or more training.")

        return 0

    except FileNotFoundError as e:
        print(f"\nError: {e}", file=sys.stderr)
        print("Please run augmentation.py first.")
        return 1
    except Exception as e:
        print(f"\nUnexpected error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
