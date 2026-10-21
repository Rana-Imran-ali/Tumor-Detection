"""
NeuroScan AI: Held-out Test Data Evaluation Pipeline
Evaluates the trained MobileNetV2 model strictly on held-out test data to prevent data leakage.

Generates:
  - Accuracy, Precision, Recall, and F1-score
  - Confusion Matrix Plot (outputs/confusion_matrix.png)
  - Full Classification Report (outputs/classification_report.txt)
  - Quantitative Metrics (outputs/metrics.json)

Usage:
    python evaluate.py --test_dir dataset/Testing --model_path models/brain_tumor_model.keras
"""

import os
import sys
import json
import argparse
import numpy as np

def evaluate_model(test_dir="dataset/Testing", model_path="models/brain_tumor_model.keras", output_dir="outputs"):
    """
    Evaluates trained model on held-out test data.
    Ensures data leakage prevention by applying deterministic preprocessing with shuffle=False.
    """
    print("=" * 70)
    print(" NeuroScan AI: Model Evaluation on Held-Out Test Data")
    print("=" * 70)
    print(f" Target Model : {model_path}")
    print(f" Test Set Dir : {test_dir}")
    print(f" Output Dir   : {output_dir}")
    print("=" * 70)

    # 1. Verification of inputs (Strictly No Fabrication)
    if not os.path.exists(model_path):
        print(f"\n[!] Error: Model file '{model_path}' not found.")
        print("    Please train the model first using: python train.py --data_dir dataset/")
        sys.exit(1)

    if not os.path.exists(test_dir):
        print(f"\n[!] Error: Held-out test directory '{test_dir}' not found.")
        print("    Ensure the dataset has been extracted with a Testing/ split.")
        print("    You can run: python download_dataset.py")
        sys.exit(1)

    try:
        import tensorflow as tf
        from tensorflow.keras.models import load_model
        from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
    except ImportError:
        print("[!] Error: TensorFlow is required. Install via: pip install tensorflow")
        sys.exit(1)

    try:
        from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support
        import matplotlib
        matplotlib.use('Agg')  # Headless rendering
        import matplotlib.pyplot as plt
    except ImportError:
        print("[!] Error: scikit-learn and matplotlib are required.")
        print("    Install via: pip install scikit-learn matplotlib")
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)

    # 2. Load held-out test data (Strictly shuffle=False to prevent ordering misalignment)
    print("\n[1/4] Loading held-out test dataset...")
    IMG_SIZE = (224, 224)
    BATCH_SIZE = 32

    test_ds = tf.keras.utils.image_dataset_from_directory(
        test_dir,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="categorical",
        shuffle=False  # CRITICAL: Preserve sequential ordering to match labels
    )

    class_names = test_ds.class_names
    num_classes = len(class_names)
    print(f"  [+] Classes identified ({num_classes}): {class_names}")

    # 3. Load trained model
    print(f"\n[2/4] Loading model checkpoint from: {model_path}...")
    model = load_model(model_path)

    # 4. Extract ground truth labels and compute predictions
    print("\n[3/4] Running inference over test set (preventing data leakage)...")
    has_preprocess = any(
        "preprocess" in getattr(layer, "name", "").lower() or "rescaling" in getattr(layer, "name", "").lower()
        for layer in getattr(model, "layers", [])
    )

    y_true_indices = []
    y_pred_probs = []

    # Iterate over dataset batches
    for images, labels in test_ds:
        processed_images = images if has_preprocess else ((tf.cast(images, tf.float32) / 127.5) - 1.0)
        preds = model.predict(processed_images, verbose=0)
        y_pred_probs.append(preds)
        y_true_indices.append(np.argmax(labels.numpy(), axis=1))

    y_true = np.concatenate(y_true_indices)
    y_pred_probs = np.concatenate(y_pred_probs)
    y_pred = np.argmax(y_pred_probs, axis=1)

    # 5. Compute Quantitative Metrics
    print("\n[4/4] Calculating evaluation metrics...")
    acc = accuracy_score(y_true, y_pred)
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(y_true, y_pred, average="macro")
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted")
    cm = confusion_matrix(y_true, y_pred)

    clf_report_str = classification_report(y_true, y_pred, target_names=class_names, digits=4)
    clf_report_dict = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)

    # Display on console
    print("\n" + "=" * 70)
    print(" HELD-OUT TEST DATA EVALUATION RESULTS")
    print("=" * 70)
    print(f" Overall Accuracy      : {acc * 100:.2f}%")
    print(f" Macro Precision       : {precision_macro * 100:.2f}%")
    print(f" Macro Recall          : {recall_macro * 100:.2f}%")
    print(f" Macro F1-Score        : {f1_macro * 100:.2f}%")
    print(f" Weighted F1-Score     : {f1_weighted * 100:.2f}%")
    print("-" * 70)
    print(" Per-Class Breakdown:\n")
    print(clf_report_str)
    print(" Confusion Matrix:")
    print(cm)
    print("=" * 70)

    # 6. Save Classification Report to outputs/
    report_file = os.path.join(output_dir, "classification_report.txt")
    with open(report_file, "w") as f:
        f.write("NeuroScan AI: MobileNetV2 Test Evaluation Report\n")
        f.write(f"Model Path : {model_path}\n")
        f.write(f"Test Set   : {test_dir}\n")
        f.write("=" * 65 + "\n\n")
        f.write(f"Overall Accuracy: {acc * 100:.2f}%\n\n")
        f.write(clf_report_str)
        f.write("\nConfusion Matrix:\n")
        f.write(np.array2string(cm))
    print(f"\n[+] Text report saved to: {report_file}")

    # 7. Save JSON Metrics to outputs/
    metrics_file = os.path.join(output_dir, "metrics.json")
    metrics_data = {
        "overall_accuracy": round(float(acc), 4),
        "macro_precision": round(float(precision_macro), 4),
        "macro_recall": round(float(recall_macro), 4),
        "macro_f1": round(float(f1_macro), 4),
        "weighted_f1": round(float(f1_weighted), 4),
        "confusion_matrix": cm.tolist(),
        "per_class_metrics": clf_report_dict
    }
    with open(metrics_file, "w") as f:
        json.dump(metrics_data, f, indent=4)
    print(f"[+] Metrics JSON saved to: {metrics_file}")

    # 8. Plot and Save Confusion Matrix Heatmap
    cm_plot_file = os.path.join(output_dir, "confusion_matrix.png")
    fig, ax = plt.subplots(figsize=(7, 6))
    
    # Plot heatmap using matplotlib
    cax = ax.matshow(cm, cmap=plt.cm.Blues, alpha=0.85)
    fig.colorbar(cax)

    # Configure axes
    ax.set_xticks(range(num_classes))
    ax.set_yticks(range(num_classes))
    ax.set_xticklabels(class_names, rotation=30, ha="left", fontsize=10)
    ax.set_yticklabels(class_names, fontsize=10)

    # Add numeric text labels in each cell
    thresh = cm.max() / 2.0
    for i in range(num_classes):
        for j in range(num_classes):
            val = cm[i, j]
            color = "white" if val > thresh else "black"
            ax.text(j, i, f"{val}", ha="center", va="center", color=color, fontsize=11, fontweight="bold")

    plt.title("NeuroScan AI: Brain Tumor Confusion Matrix", pad=20, fontsize=12, fontweight="bold")
    plt.xlabel("Predicted Class", labelpad=10, fontsize=11)
    plt.ylabel("True Class", labelpad=10, fontsize=11)
    plt.tight_layout()
    plt.savefig(cm_plot_file, dpi=300)
    plt.close()
    print(f"[+] Confusion matrix plot saved to: {cm_plot_file}")
    print("\nAll evaluation artifacts successfully written to outputs/.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate MobileNetV2 on held-out test data")
    parser.add_argument("--test_dir", type=str, default="dataset/Testing", help="Path to held-out test directory")
    parser.add_argument("--model_path", type=str, default="models/brain_tumor_model.keras", help="Path to trained .keras model")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Directory to save evaluation plots and reports")
    args = parser.parse_args()

    evaluate_model(
        test_dir=args.test_dir,
        model_path=args.model_path,
        output_dir=args.output_dir
    )
