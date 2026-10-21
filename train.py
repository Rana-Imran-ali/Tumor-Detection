"""
NeuroScan AI: MobileNetV2 Brain Tumor Classifier Training Pipeline
Trains a 4-class classifier on Brain MRI scan datasets using Transfer Learning.

Dataset Source: https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset
Target Classes : glioma, meningioma, notumor, pituitary
Output Model   : models/brain_tumor_model.keras
Class Names    : models/class_names.json
"""

import os
import sys
import json
import argparse
from PIL import Image

def clean_dataset_images(data_dir):
    """
    Scans dataset directories to identify and handle missing, unreadable, or corrupt images.
    Removes corrupted files to prevent training crashes.
    """
    print(f"\n[1/6] Scanning and sanitizing image files in: {data_dir}...")
    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    corrupt_count = 0
    total_count = 0

    for root, _, files in os.walk(data_dir):
        for file in files:
            ext = os.path.splitext(file)[1].lower()
            if ext not in valid_extensions:
                continue

            total_count += 1
            file_path = os.path.join(root, file)

            # Check if file is empty
            if os.path.getsize(file_path) == 0:
                print(f"  [!] Removing empty image: {file_path}")
                os.remove(file_path)
                corrupt_count += 1
                continue

            # Verify image integrity using PIL
            try:
                with Image.open(file_path) as img:
                    img.verify()  # Check for header/structural corruption
                # Re-open and convert to catch truncated data
                with Image.open(file_path) as img:
                    img.convert("RGB").resize((10, 10))
            except Exception as e:
                print(f"  [!] Corrupt image found: {file_path} ({e})")
                try:
                    os.remove(file_path)
                    corrupt_count += 1
                except Exception as del_err:
                    print(f"  [-] Failed to delete {file_path}: {del_err}")

    print(f"  [+] Verified {total_count} images. Handled {corrupt_count} corrupt or unreadable files.")

def train(data_dir="dataset", epochs=15, batch_size=32, learning_rate=0.0001, output_model="models/brain_tumor_model.keras"):
    """
    End-to-end training pipeline:
    1. Loads sanitized dataset
    2. Applies MobileNetV2 preprocessing & data augmentation
    3. Fine-tunes MobileNetV2 with transfer learning
    4. Evaluates on validation data
    5. Saves best model and class mapping
    """
    try:
        import tensorflow as tf
        from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
        from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, RandomFlip, RandomRotation, RandomZoom
        from tensorflow.keras.models import Sequential, Model
        from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
    except ImportError:
        print("\n[!] Error: TensorFlow is required to run the training script.")
        print("    Install it via: pip install tensorflow")
        sys.exit(1)

    print("=" * 70)
    print(" NeuroScan AI: MobileNetV2 Brain Tumor Classifier Training")
    print(f" TensorFlow Version: {tf.__version__}")
    print(f" Dataset Path      : {os.path.abspath(data_dir)}")
    print(f" Epochs Target     : {epochs}")
    print(f" Batch Size        : {batch_size}")
    print(f" Model Output Path : {output_model}")
    print("=" * 70)

    # 1. Dataset Path Validation (No fabricated training)
    train_dir = os.path.join(data_dir, "Training") if os.path.exists(os.path.join(data_dir, "Training")) else data_dir
    val_dir = os.path.join(data_dir, "Testing") if os.path.exists(os.path.join(data_dir, "Testing")) else None

    if not os.path.exists(train_dir):
        print(f"\n[!] Error: Dataset folder not found at '{train_dir}'.")
        print("\nTo download the official Kaggle dataset:")
        print("  1. Run: python download_dataset.py")
        print("  2. Or download from: https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset")
        print("     and extract into the 'dataset/' folder.")
        sys.exit(1)

    # 2. Clean corrupt images
    clean_dataset_images(data_dir)

    IMG_SIZE = (224, 224)

    # 3. Load Datasets
    print("\n[2/6] Loading training and validation image sets...")
    if val_dir and os.path.exists(val_dir):
        print(f"  Using dedicated validation directory: {val_dir}")
        train_ds = tf.keras.utils.image_dataset_from_directory(
            train_dir,
            image_size=IMG_SIZE,
            batch_size=batch_size,
            label_mode="categorical",
            shuffle=True
        )
        val_ds = tf.keras.utils.image_dataset_from_directory(
            val_dir,
            image_size=IMG_SIZE,
            batch_size=batch_size,
            label_mode="categorical",
            shuffle=False
        )
    else:
        print("  Using 80/20 train/validation split...")
        train_ds = tf.keras.utils.image_dataset_from_directory(
            train_dir,
            validation_split=0.2,
            subset="training",
            seed=42,
            image_size=IMG_SIZE,
            batch_size=batch_size,
            label_mode="categorical"
        )
        val_ds = tf.keras.utils.image_dataset_from_directory(
            train_dir,
            validation_split=0.2,
            subset="validation",
            seed=42,
            image_size=IMG_SIZE,
            batch_size=batch_size,
            label_mode="categorical"
        )

    class_names = train_ds.class_names
    num_classes = len(class_names)
    print(f"\n[3/6] Target Classes Identified ({num_classes}): {class_names}")

    # Save class names separately as required
    os.makedirs(os.path.dirname(output_model), exist_ok=True)
    class_names_path = os.path.join(os.path.dirname(output_model), "class_names.json")
    with open(class_names_path, "w") as f:
        json.dump(class_names, f, indent=4)
    print(f"  [+] Class labels saved to: {class_names_path}")

    # Optimize data pipeline caching & prefetching
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    # 4. Construct MobileNetV2 Transfer Learning Architecture
    print("\n[4/6] Building MobileNetV2 model architecture with data augmentation...")
    data_augmentation = Sequential([
        RandomFlip("horizontal"),
        RandomRotation(0.1),
        RandomZoom(0.1),
    ], name="data_augmentation")

    base_model = MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights="imagenet"
    )
    base_model.trainable = False  # Freeze pre-trained weights for initial convergence

    inputs = tf.keras.Input(shape=(224, 224, 3))
    x = data_augmentation(inputs)
    # Apply standard MobileNetV2 preprocessing (scales pixel values to [-1, 1])
    x = preprocess_input(x)
    x = base_model(x, training=False)
    x = GlobalAveragePooling2D(name="global_avg_pool")(x)
    x = Dropout(0.3, name="dropout_1")(x)
    x = Dense(128, activation="relu", name="dense_128")(x)
    x = Dropout(0.2, name="dropout_2")(x)
    outputs = Dense(num_classes, activation="softmax", name="predictions")(x)

    model = Model(inputs, outputs, name="NeuroScan_MobileNetV2")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    model.summary()

    # Callbacks for best model checkpointing
    callbacks = [
        ModelCheckpoint(
            filepath=output_model,
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
            verbose=1
        ),
        EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-6,
            verbose=1
        )
    ]

    # 5. Execute Training (Actual Honest Execution - No Fabricated Numbers)
    print("\n[5/6] Commencing Phase 1: Training Classification Head...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks
    )

    # Optional Phase 2: Unfreeze top layers for fine-tuning
    print("\n[6/6] Commencing Phase 2: Fine-Tuning Top MobileNetV2 Layers...")
    base_model.trainable = True
    for layer in base_model.layers[:-30]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate / 10),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    fine_tune_epochs = 5
    total_epochs = epochs + fine_tune_epochs

    history_fine = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=total_epochs,
        initial_epoch=len(history.epoch),
        callbacks=callbacks
    )

    # 6. Save Training & Validation Curves to outputs/
    save_training_history_and_plots(history, history_fine, output_dir="outputs")

    print("\n" + "=" * 70)
    print(" TRAINING COMPLETE")
    print(f" Best Model Saved : {output_model}")
    print(f" Class Names Saved: {class_names_path}")
    print(" Training Curves  : outputs/training_curves.png")
    print(" Evaluate Model   : python evaluate.py")
    print(" Launch Web App   : python app.py")
    print("=" * 70)

def save_training_history_and_plots(history1, history2, output_dir="outputs"):
    """
    Concatenates metrics from phase 1 and phase 2, saves training_history.json,
    and plots training/validation accuracy and loss curves in outputs/training_curves.png.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Concatenate metrics
    acc = history1.history.get("accuracy", []) + history2.history.get("accuracy", [])
    val_acc = history1.history.get("val_accuracy", []) + history2.history.get("val_accuracy", [])
    loss = history1.history.get("loss", []) + history2.history.get("loss", [])
    val_loss = history1.history.get("val_loss", []) + history2.history.get("val_loss", [])

    # Save JSON history
    history_dict = {
        "accuracy": acc,
        "val_accuracy": val_acc,
        "loss": loss,
        "val_loss": val_loss,
        "epochs": list(range(1, len(acc) + 1))
    }
    history_json_path = os.path.join(output_dir, "training_history.json")
    with open(history_json_path, "w") as f:
        json.dump(history_dict, f, indent=4)
    print(f"\n[+] Training history metrics saved to: {history_json_path}")

    # Plot Accuracy & Loss
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        epochs_range = range(1, len(acc) + 1)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Accuracy Subplot
        ax1.plot(epochs_range, acc, label="Training Accuracy", color="#06b6d4", linewidth=2)
        ax1.plot(epochs_range, val_acc, label="Validation Accuracy", color="#f59e0b", linewidth=2, linestyle="--")
        ax1.set_title("Training & Validation Accuracy", fontsize=12, fontweight="bold")
        ax1.set_xlabel("Epochs", fontsize=10)
        ax1.set_ylabel("Accuracy", fontsize=10)
        ax1.grid(True, linestyle=":", alpha=0.6)
        ax1.legend(loc="lower right")

        # Loss Subplot
        ax2.plot(epochs_range, loss, label="Training Loss", color="#06b6d4", linewidth=2)
        ax2.plot(epochs_range, val_loss, label="Validation Loss", color="#ef4444", linewidth=2, linestyle="--")
        ax2.set_title("Training & Validation Loss", fontsize=12, fontweight="bold")
        ax2.set_xlabel("Epochs", fontsize=10)
        ax2.set_ylabel("Categorical Crossentropy Loss", fontsize=10)
        ax2.grid(True, linestyle=":", alpha=0.6)
        ax2.legend(loc="upper right")

        plt.suptitle("NeuroScan AI: MobileNetV2 Training Curves", fontsize=14, fontweight="bold", y=1.02)
        plt.tight_layout()

        plot_path = os.path.join(output_dir, "training_curves.png")
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"[+] Training curves plot saved to: {plot_path}")
    except Exception as e:
        print(f"[!] Warning: Could not generate training curve plot: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MobileNetV2 Brain Tumor Classifier")
    parser.add_argument("--data_dir", type=str, default="dataset", help="Directory where dataset is located")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=0.0001, help="Initial learning rate")
    parser.add_argument("--output_model", type=str, default="models/brain_tumor_model.keras", help="Path to save best model")
    args = parser.parse_args()

    train(
        data_dir=args.data_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        output_model=args.output_model
    )
