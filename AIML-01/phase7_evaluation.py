"""
PHASE 7: INDEPENDENT MODEL EVALUATION

Evaluates the best Phase 6 checkpoint on the untouched test set.

Classes:
    0 = NORMAL
    1 = PNEUMONIA

Does not train or modify the model.
"""

import argparse
import csv
import json

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from phase4_preprocessing import Phase4Pipeline
from phase5_model import ChestXRayCNN


# ============================================================
# 1. PATHS AND CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "AIML-01/data/processed/phase6/checkpoints/best_model.pt"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "AIML-01/data/processed/phase7"
)

CLASS_NAMES = [
    "NORMAL",
    "PNEUMONIA",
]

CLASS_IDS = [0, 1]

# Fixed threshold: selected before inspecting test results.
PNEUMONIA_THRESHOLD = 0.50


# ============================================================
# 2. DEVICE
# ============================================================

def get_device():

    if torch.cuda.is_available():

        device = torch.device("cuda")

        print(
            "GPU:",
            torch.cuda.get_device_name(device),
        )

    else:

        device = torch.device("cpu")

        print("CUDA unavailable. Using CPU.")

    return device


# ============================================================
# 3. LOAD BEST CHECKPOINT
# ============================================================

def load_model(device):

    if not CHECKPOINT_PATH.exists():

        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT_PATH}"
        )

    print("\n========== LOADING BEST CHECKPOINT ==========")

    try:

        checkpoint = torch.load(
            CHECKPOINT_PATH,
            map_location=device,
            weights_only=False,
        )

    except TypeError:

        checkpoint = torch.load(
            CHECKPOINT_PATH,
            map_location=device,
        )

    if "model_state_dict" not in checkpoint:

        raise ValueError(
            "Checkpoint does not contain model_state_dict."
        )

    if checkpoint.get("class_names") != CLASS_NAMES:

        raise ValueError(
            "Checkpoint class names do not match evaluation classes."
        )

    model = ChestXRayCNN()

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model = model.to(device)

    model.eval()

    print("Checkpoint:", CHECKPOINT_PATH)

    print("Best checkpoint epoch:", checkpoint["epoch"])

    print(
        "Best validation loss:",
        checkpoint["best_val_loss"],
    )

    print(
        "Best validation macro F1:",
        checkpoint["best_val_f1"],
    )

    return model, checkpoint


# ============================================================
# 4. LOAD TEST DATA
# ============================================================

def load_test_dataloader(batch_size, workers):

    print("\n========== LOADING TEST DATA ==========")

    pipeline = Phase4Pipeline()

    test_loader = pipeline.create_dataloader(
        split="test",
        batch_size=batch_size,
        workers=workers,
    )

    print("Test batches:", len(test_loader))

    return test_loader


# ============================================================
# 5. MODEL INFERENCE
# ============================================================

def predict(model, test_loader, device):

    print("\n========== RUNNING TEST INFERENCE ==========")

    all_labels = []
    all_probabilities = []
    all_predictions = []
    all_paths = []

    with torch.inference_mode():

        for batch_index, batch in enumerate(test_loader):

            if isinstance(batch, dict):

                images = batch["image"]
                labels = batch["label"]
                paths = batch["path"]

            else:

                images, labels = batch[:2]

                paths = [
                    ""
                ] * len(labels)

            images = images.to(
                device,
                non_blocking=True,
            )

            labels = labels.to(
                device,
                non_blocking=True,
            )

            # Raw model outputs
            logits = model(images)

            # Convert logits to probabilities
            probabilities = torch.softmax(
                logits,
                dim=1,
            )

            # Extract PNEUMONIA probability
            pneumonia_probabilities = probabilities[:, 1]

            # Fixed threshold
            predictions = (
                pneumonia_probabilities >= PNEUMONIA_THRESHOLD
            ).long()

            all_labels.extend(
                labels.cpu().numpy().tolist()
            )

            all_probabilities.extend(
                pneumonia_probabilities.cpu().numpy().tolist()
            )

            all_predictions.extend(
                predictions.cpu().numpy().tolist()
            )

            all_paths.extend(list(paths))

            print(
                f"Processed batch "
                f"{batch_index + 1}/{len(test_loader)}"
            )

    y_true = np.asarray(
        all_labels,
        dtype=np.int64,
    )

    y_pred = np.asarray(
        all_predictions,
        dtype=np.int64,
    )

    y_probability = np.asarray(
        all_probabilities,
        dtype=np.float64,
    )

    if not (
        len(y_true)
        == len(y_pred)
        == len(y_probability)
        == len(all_paths)
    ):

        raise RuntimeError(
            "Prediction arrays have inconsistent lengths."
        )

    if len(y_true) == 0:

        raise RuntimeError(
            "Test dataset is empty."
        )

    if not np.isfinite(y_probability).all():

        raise RuntimeError(
            "Non-finite prediction probabilities detected."
        )

    if not np.isin(y_true, CLASS_IDS).all():

        raise RuntimeError(
            "Unexpected test labels detected."
        )

    return (
        y_true,
        y_pred,
        y_probability,
        all_paths,
    )


# ============================================================
# 6. METRIC CALCULATION
# ============================================================

def calculate_metrics(
    y_true,
    y_pred,
    y_probability,
):

    print("\n========== CALCULATING METRICS ==========")

    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=CLASS_IDS,
    )

    tn, fp, fn, tp = matrix.ravel()

    accuracy = accuracy_score(
        y_true,
        y_pred,
    )

    precision = precision_score(
        y_true,
        y_pred,
        average="binary",
        pos_label=1,
        zero_division=0,
    )

    recall = recall_score(
        y_true,
        y_pred,
        average="binary",
        pos_label=1,
        zero_division=0,
    )

    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else 0.0
    )

    macro_precision = precision_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )

    macro_recall = recall_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )

    macro_f1 = f1_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )

    weighted_f1 = f1_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0,
    )

    balanced_accuracy = balanced_accuracy_score(
        y_true,
        y_pred,
    )

    # ROC-AUC requires both classes in the test set.
    if len(np.unique(y_true)) == 2:

        roc_auc = roc_auc_score(
            y_true,
            y_probability,
        )

        pr_auc = average_precision_score(
            y_true,
            y_probability,
        )

        fpr, tpr, roc_thresholds = roc_curve(
            y_true,
            y_probability,
        )

        pr_precision, pr_recall, pr_thresholds = (
            precision_recall_curve(
                y_true,
                y_probability,
            )
        )

    else:

        roc_auc = None
        pr_auc = None

        fpr = []
        tpr = []
        roc_thresholds = []

        pr_precision = []
        pr_recall = []
        pr_thresholds = []

    report = classification_report(
        y_true,
        y_pred,
        labels=CLASS_IDS,
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )

    metrics = {

        "test_samples": int(len(y_true)),

        "threshold": PNEUMONIA_THRESHOLD,

        "accuracy": float(accuracy),

        "balanced_accuracy": float(balanced_accuracy),

        "pneumonia_precision": float(precision),

        "pneumonia_recall_sensitivity": float(recall),

        "normal_specificity": float(specificity),

        "macro_precision": float(macro_precision),

        "macro_recall": float(macro_recall),

        "macro_f1": float(macro_f1),

        "weighted_f1": float(weighted_f1),

        "roc_auc": (
            float(roc_auc)
            if roc_auc is not None
            else None
        ),

        "pr_auc_average_precision": (
            float(pr_auc)
            if pr_auc is not None
            else None
        ),

        "confusion_matrix": matrix.tolist(),

        "confusion_matrix_labels": CLASS_NAMES,

        "true_negative": int(tn),

        "false_positive": int(fp),

        "false_negative": int(fn),

        "true_positive": int(tp),

        "classification_report": report,

        "roc_curve": {
            "false_positive_rate": fpr.tolist(),
            "true_positive_rate": tpr.tolist(),
            "thresholds": [
                  float(value) if np.isfinite(value) else None
                  for value in roc_thresholds
            ],
        },

        "precision_recall_curve": {
            "precision": pr_precision.tolist(),
            "recall": pr_recall.tolist(),
            "thresholds": [
                 float(value) if np.isfinite(value) else None
                 for value in pr_thresholds
            ],
        },
    }

    return metrics


# ============================================================
# 7. SAVE PREDICTIONS
# ============================================================

def save_predictions(
    paths,
    y_true,
    y_pred,
    y_probability,
):

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = OUTPUT_DIR / "test_predictions.csv"

    with open(
        output_path,
        "w",
        newline="",
    ) as file:

        writer = csv.writer(file)

        writer.writerow([
            "image_path",
            "actual_label",
            "predicted_label",
            "pneumonia_probability",
            "threshold",
            "correct",
        ])

        for path, actual, predicted, probability in zip(
            paths,
            y_true,
            y_pred,
            y_probability,
        ):

            writer.writerow([
                path,
                CLASS_NAMES[int(actual)],
                CLASS_NAMES[int(predicted)],
                f"{float(probability):.8f}",
                PNEUMONIA_THRESHOLD,
                bool(actual == predicted),
            ])

    print("Predictions saved:", output_path)


# ============================================================
# 8. SAVE REPORT
# ============================================================

def save_report(metrics, checkpoint):

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    report = {

        "phase": 7,

        "name": "Independent Model Evaluation",

        "created_at": datetime.now(
            timezone.utc
        ).isoformat(),

        "checkpoint": str(CHECKPOINT_PATH),

        "checkpoint_epoch": int(checkpoint["epoch"]),

        "evaluation_split": "test",

        "test_used_for_training": False,

        "test_used_for_model_selection": False,

        "class_names": CLASS_NAMES,

        "threshold": PNEUMONIA_THRESHOLD,

        "metrics": metrics,

        "limitations": [
            "The dataset split is image-level, not patient-level.",
            "The dataset contains pediatric chest X-rays.",
            "Results do not establish clinical validity.",
            "A single test evaluation does not establish external generalization.",
        ],

        "status": "evaluation_completed",
    }

    output_path = OUTPUT_DIR / "phase7_report.json"

    with open(output_path, "w") as file:

        json.dump(
            report,
            file,
            indent=4,
            allow_nan=False,
        )

    print("Report saved:", output_path)


# ============================================================
# 9. PRINT SUMMARY
# ============================================================

def print_summary(metrics):

    print("\n" + "=" * 55)

    print("PHASE 7 — TEST EVALUATION RESULTS")

    print("=" * 55)

    print("Test samples:", metrics["test_samples"])

    print(
        f"Accuracy: "
        f"{metrics['accuracy']:.4f}"
    )

    print(
        f"Balanced accuracy: "
        f"{metrics['balanced_accuracy']:.4f}"
    )

    print(
        f"Pneumonia precision: "
        f"{metrics['pneumonia_precision']:.4f}"
    )

    print(
        f"Pneumonia sensitivity/recall: "
        f"{metrics['pneumonia_recall_sensitivity']:.4f}"
    )

    print(
        f"Normal specificity: "
        f"{metrics['normal_specificity']:.4f}"
    )

    print(
        f"Macro F1: "
        f"{metrics['macro_f1']:.4f}"
    )

    print(
        f"Weighted F1: "
        f"{metrics['weighted_f1']:.4f}"
    )

    print(
        f"ROC-AUC: "
        f"{metrics['roc_auc']}"
    )

    print(
        f"PR-AUC: "
        f"{metrics['pr_auc_average_precision']}"
    )

    print("\nConfusion matrix:")

    print("Rows = Actual, Columns = Predicted")

    print("              NORMAL  PNEUMONIA")

    matrix = metrics["confusion_matrix"]

    print(
        f"NORMAL       {matrix[0][0]:6d}  {matrix[0][1]:9d}"
    )

    print(
        f"PNEUMONIA    {matrix[1][0]:6d}  {matrix[1][1]:9d}"
    )

    print("\nClassification report:")

    for class_name in CLASS_NAMES:

        values = metrics["classification_report"][class_name]

        print(
            f"{class_name}: "
            f"precision={values['precision']:.4f}, "
            f"recall={values['recall']:.4f}, "
            f"f1={values['f1-score']:.4f}, "
            f"support={values['support']}"
        )

    print("\n" + "=" * 55)


# ============================================================
# 10. MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description="Phase 7: Independent CNN Evaluation"
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=2,
    )

    args = parser.parse_args()

    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")

    if args.workers < 0:
        parser.error("--workers cannot be negative")

    device = get_device()

    model, checkpoint = load_model(device)

    test_loader = load_test_dataloader(
        batch_size=args.batch_size,
        workers=args.workers,
    )

    y_true, y_pred, y_probability, paths = predict(
        model,
        test_loader,
        device,
    )

    metrics = calculate_metrics(
        y_true,
        y_pred,
        y_probability,
    )

    save_predictions(
        paths,
        y_true,
        y_pred,
        y_probability,
    )

    save_report(
        metrics,
        checkpoint,
    )

    print_summary(metrics)

    print("\nPHASE 7 EVALUATION COMPLETED.")


if __name__ == "__main__":
    main()
