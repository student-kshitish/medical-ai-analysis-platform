"""
PHASE 6: MODEL TRAINING

Medical AI Analysis Platform
Task: Pediatric chest X-ray classification

Classes:
    0 = NORMAL
    1 = PNEUMONIA

Uses:
    Phase 4: MedicalImageDataset / Phase4Pipeline
    Phase 5: ChestXRayCNN

Important:
    The test dataset is not used during training.
"""

import argparse
import csv
import json
import random
import time

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

from phase4_preprocessing import Phase4Pipeline
from phase5_model import ChestXRayCNN


# ============================================================
# 1. CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

TRAIN_MANIFEST = (
    PROJECT_ROOT
    / "AIML-01/data/processed/phase2/train_manifest.csv"
)

VAL_MANIFEST = (
    PROJECT_ROOT
    / "AIML-01/data/processed/phase2/val_manifest.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "AIML-01/data/processed/phase6"
)

CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"

HISTORY_PATH = OUTPUT_DIR / "training_history.json"

CLASS_NAMES = [
    "NORMAL",
    "PNEUMONIA",
]

CLASS_TO_INDEX = {
    "NORMAL": 0,
    "PNEUMONIA": 1,
}


class TrainingConfig:

    EPOCHS = 20
    BATCH_SIZE = 16
    WORKERS = 2

    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-4

    SCHEDULER_FACTOR = 0.5
    SCHEDULER_PATIENCE = 2

    EARLY_STOPPING_PATIENCE = 5

    SEED = 42

    GRADIENT_CLIP = 1.0


# ============================================================
# 2. REPRODUCIBILITY
# ============================================================

def set_seed(seed):

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# 3. DEVICE SELECTION
# ============================================================

def get_device():

    if torch.cuda.is_available():

        device = torch.device("cuda")

        print(
            "GPU detected:",
            torch.cuda.get_device_name(device),
        )

        return device

    print("CUDA unavailable. Using CPU.")

    return torch.device("cpu")


# ============================================================
# 4. CLASS WEIGHT CALCULATION
# ============================================================

def calculate_class_weights(manifest_path):
    """
    Calculate inverse-frequency weights from TRAIN only.

    Formula:

        weight[class] = total_samples /
                        (number_of_classes * class_count)

    This gives more importance to the minority class.
    """

    counts = {
        "NORMAL": 0,
        "PNEUMONIA": 0,
    }

    with open(manifest_path, newline="") as file:

        reader = csv.DictReader(file)

        for row in reader:

            label = row["label"].strip().upper()

            if label not in counts:
                raise ValueError(
                    f"Unknown label in manifest: {label}"
                )

            counts[label] += 1

    total = sum(counts.values())

    if total == 0:
        raise ValueError("Training manifest is empty.")

    weights = []

    for class_name in CLASS_NAMES:

        count = counts[class_name]

        if count == 0:
            raise ValueError(
                f"No training samples for {class_name}"
            )

        weight = total / (len(CLASS_NAMES) * count)

        weights.append(weight)

    weights_tensor = torch.tensor(
        weights,
        dtype=torch.float32,
    )

    print("\n========== CLASS DISTRIBUTION ==========")

    for class_name in CLASS_NAMES:

        print(
            f"{class_name}: {counts[class_name]}"
        )

    print("\nClass weights:")

    for index, class_name in enumerate(CLASS_NAMES):

        print(
            f"{class_name}: "
            f"{weights_tensor[index].item():.6f}"
        )

    return weights_tensor


# ============================================================
# 5. METRIC CALCULATION
# ============================================================

def calculate_metrics(confusion_matrix):
    """
    Calculate accuracy, precision, recall and F1.

    Confusion matrix layout:

                  Predicted
                  NORMAL  PNEUMONIA

    Actual NORMAL
    Actual PNEUMONIA
    """

    matrix = confusion_matrix.astype(np.float64)

    total = matrix.sum()

    if total == 0:
        raise ValueError("No samples available for metrics.")

    accuracy = np.trace(matrix) / total

    precisions = []
    recalls = []
    f1_scores = []

    for index in range(len(CLASS_NAMES)):

        true_positive = matrix[index, index]

        false_positive = (
            matrix[:, index].sum() - true_positive
        )

        false_negative = (
            matrix[index, :].sum() - true_positive
        )

        precision = (
            true_positive / (true_positive + false_positive)
            if true_positive + false_positive > 0
            else 0.0
        )

        recall = (
            true_positive / (true_positive + false_negative)
            if true_positive + false_negative > 0
            else 0.0
        )

        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall > 0
            else 0.0
        )

        precisions.append(float(precision))
        recalls.append(float(recall))
        f1_scores.append(float(f1))

    return {
        "accuracy": float(accuracy),
        "macro_precision": float(np.mean(precisions)),
        "macro_recall": float(np.mean(recalls)),
        "macro_f1": float(np.mean(f1_scores)),
        "class_precision": precisions,
        "class_recall": recalls,
        "class_f1": f1_scores,
        "confusion_matrix": matrix.astype(int).tolist(),
    }


# ============================================================
# 6. BATCH EXTRACTION
# ============================================================

def unpack_batch(batch):
    """
    Support dictionary batches and tuple/list batches.
    """

    if isinstance(batch, dict):

        images = batch["image"]
        labels = batch["label"]

    elif isinstance(batch, (tuple, list)):

        images = batch[0]
        labels = batch[1]

    else:

        raise TypeError(
            f"Unsupported batch type: {type(batch)}"
        )

    return images, labels


# ============================================================
# 7. TRAINING / VALIDATION ENGINE
# ============================================================

def run_epoch(
    model,
    dataloader,
    criterion,
    device,
    optimizer=None,
    training=False,
):
    """
    Run one complete training or validation epoch.

    Training:
        Forward -> Loss -> Backward -> Optimizer

    Validation:
        Forward -> Loss -> Metrics
    """

    if training:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_samples = 0

    confusion_matrix = np.zeros(
        (2, 2),
        dtype=np.int64,
    )

    start_time = time.time()

    context = torch.enable_grad if training else torch.no_grad

    with context():

        for batch_index, batch in enumerate(dataloader):

            images, labels = unpack_batch(batch)

            images = images.to(
                device,
                non_blocking=True,
            )

            labels = labels.to(
                device,
                non_blocking=True,
            )

            if training:

                optimizer.zero_grad(set_to_none=True)

            # Forward propagation
            outputs = model(images)

            # Calculate classification loss
            loss = criterion(outputs, labels)

            if training:

                # Backpropagation
                loss.backward()

                # Prevent unusually large gradients
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=TrainingConfig.GRADIENT_CLIP,
                )

                # Update model parameters
                optimizer.step()

            predictions = torch.argmax(
                outputs,
                dim=1,
            )

            batch_size = labels.size(0)

            total_loss += loss.item() * batch_size

            total_samples += batch_size

            # Update confusion matrix
            for actual, predicted in zip(
                labels.detach().cpu().numpy(),
                predictions.detach().cpu().numpy(),
            ):

                confusion_matrix[
                    int(actual),
                    int(predicted),
                ] += 1

    if total_samples == 0:
        raise RuntimeError("DataLoader returned no samples.")

    average_loss = total_loss / total_samples

    metrics = calculate_metrics(confusion_matrix)

    elapsed = time.time() - start_time

    return {
        "loss": float(average_loss),
        **metrics,
        "samples": int(total_samples),
        "seconds": round(elapsed, 2),
    }


# ============================================================
# 8. CHECKPOINT SAVING
# ============================================================

def save_checkpoint(
    path,
    model,
    optimizer,
    scheduler,
    epoch,
    best_val_loss,
    best_val_f1,
    patience_counter,
    history,
):
    """
    Save everything needed to resume training.
    """

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint = {

        "epoch": epoch,

        "model_state_dict": model.state_dict(),

        "optimizer_state_dict": optimizer.state_dict(),

        "scheduler_state_dict": scheduler.state_dict(),

        "best_val_loss": best_val_loss,

        "best_val_f1": best_val_f1,

        "patience_counter": patience_counter,

        "history": history,

        "class_names": CLASS_NAMES,

        "config": {
            "epochs": TrainingConfig.EPOCHS,
            "batch_size": TrainingConfig.BATCH_SIZE,
            "learning_rate": TrainingConfig.LEARNING_RATE,
            "weight_decay": TrainingConfig.WEIGHT_DECAY,
        },
    }

    temporary_path = path.with_suffix(
        path.suffix + ".tmp"
    )

    torch.save(
        checkpoint,
        temporary_path,
    )

    temporary_path.replace(path)


# ============================================================
# 9. CHECKPOINT LOADING
# ============================================================

def load_checkpoint(
    path,
    model,
    optimizer,
    scheduler,
    device,
):
    """
    Restore model and optimizer state.

    Only load checkpoints produced by this project
    or otherwise fully trusted sources.
    """

    try:

        checkpoint = torch.load(
            path,
            map_location=device,
            weights_only=False,
        )

    except TypeError:

        # Compatibility with older PyTorch versions
        checkpoint = torch.load(
            path,
            map_location=device,
        )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    optimizer.load_state_dict(
        checkpoint["optimizer_state_dict"]
    )

    scheduler.load_state_dict(
        checkpoint["scheduler_state_dict"]
    )

    print(
        f"Resuming from epoch {checkpoint['epoch'] + 1}"
    )

    return (
        checkpoint["epoch"] + 1,
        checkpoint["best_val_loss"],
        checkpoint["best_val_f1"],
        checkpoint["patience_counter"],
        checkpoint["history"],
    )


# ============================================================
# 10. SAVE TRAINING HISTORY
# ============================================================

def save_history(history):

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(HISTORY_PATH, "w") as file:

        json.dump(
            history,
            file,
            indent=4,
        )


# ============================================================
# 11. MAIN TRAINING FUNCTION
# ============================================================

def train_model(
    epochs,
    batch_size,
    workers,
    learning_rate,
    patience,
    resume_path=None,
):

    set_seed(TrainingConfig.SEED)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    CHECKPOINT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    device = get_device()

    # --------------------------------------------------------
    # Load Phase 4 preprocessing
    # --------------------------------------------------------

    print("\n========== LOADING PHASE 4 ==========")

    pipeline = Phase4Pipeline()

    train_loader = pipeline.create_dataloader(
        split="train",
        batch_size=batch_size,
        workers=workers,
    )

    val_loader = pipeline.create_dataloader(
        split="val",
        batch_size=batch_size,
        workers=workers,
    )

    print(
        "Training batches:",
        len(train_loader),
    )

    print(
        "Validation batches:",
        len(val_loader),
    )

    # --------------------------------------------------------
    # Load Phase 5 CNN
    # --------------------------------------------------------

    print("\n========== LOADING PHASE 5 MODEL ==========")

    model = ChestXRayCNN().to(device)

    print(
        "Trainable parameters:",
        sum(
            p.numel()
            for p in model.parameters()
            if p.requires_grad
        ),
    )

    # --------------------------------------------------------
    # Calculate class weights from TRAIN only
    # --------------------------------------------------------

    class_weights = calculate_class_weights(
        TRAIN_MANIFEST
    ).to(device)

    criterion = nn.CrossEntropyLoss(
        weight=class_weights
    )

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    optimizer = AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=TrainingConfig.WEIGHT_DECAY,
    )

    # --------------------------------------------------------
    # Learning rate scheduler
    # --------------------------------------------------------

    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=TrainingConfig.SCHEDULER_FACTOR,
        patience=TrainingConfig.SCHEDULER_PATIENCE,
        min_lr=1e-7,
    )

    # --------------------------------------------------------
    # Training state
    # --------------------------------------------------------

    best_val_loss = float("inf")
    best_val_f1 = 0.0

    patience_counter = 0

    start_epoch = 1

    history = []

    if resume_path:

        start_epoch, best_val_loss, best_val_f1, patience_counter, history = (
            load_checkpoint(
                Path(resume_path),
                model,
                optimizer,
                scheduler,
                device,
            )
        )

    # --------------------------------------------------------
    # Training loop
    # --------------------------------------------------------

    print("\n========== STARTING TRAINING ==========")

    print("Epochs:", epochs)
    print("Batch size:", batch_size)
    print("Learning rate:", learning_rate)
    print("Early stopping patience:", patience)

    for epoch in range(start_epoch, epochs + 1):

        print("\n" + "=" * 60)

        print(f"EPOCH {epoch}/{epochs}")

        print("=" * 60)

        # Training
        train_results = run_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            device=device,
            optimizer=optimizer,
            training=True,
        )

        # Validation
        val_results = run_epoch(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
            training=False,
        )

        # Update scheduler using validation loss
        scheduler.step(val_results["loss"])

        current_lr = optimizer.param_groups[0]["lr"]

        print("\nTRAINING RESULTS")

        print(
            f"Loss: {train_results['loss']:.4f}"
        )

        print(
            f"Accuracy: {train_results['accuracy']:.4f}"
        )

        print(
            f"Macro F1: {train_results['macro_f1']:.4f}"
        )

        print("\nVALIDATION RESULTS")

        print(
            f"Loss: {val_results['loss']:.4f}"
        )

        print(
            f"Accuracy: {val_results['accuracy']:.4f}"
        )

        print(
            f"Macro Precision: {val_results['macro_precision']:.4f}"
        )

        print(
            f"Macro Recall: {val_results['macro_recall']:.4f}"
        )

        print(
            f"Macro F1: {val_results['macro_f1']:.4f}"
        )

        print(
            "Learning rate:",
            f"{current_lr:.8f}",
        )

        # ----------------------------------------------------
        # Save history
        # ----------------------------------------------------

        epoch_record = {
            "epoch": epoch,
            "learning_rate": current_lr,
            "train": train_results,
            "validation": val_results,
        }

        history.append(epoch_record)

        save_history(history)

        # ----------------------------------------------------
        # Determine improvement
        # ----------------------------------------------------

        improved = (
            val_results["loss"] < best_val_loss
        )

        if improved:

            best_val_loss = val_results["loss"]

            best_val_f1 = val_results["macro_f1"]

            patience_counter = 0

            best_path = CHECKPOINT_DIR / "best_model.pt"

            save_checkpoint(
                best_path,
                model,
                optimizer,
                scheduler,
                epoch,
                best_val_loss,
                best_val_f1,
                patience_counter,
                history,
            )

            print("\nBEST MODEL UPDATED:", best_path)

        else:

            patience_counter += 1

            print(
                "\nNo validation-loss improvement:",
                f"{patience_counter}/{patience}",
            )

        # ----------------------------------------------------
        # Save latest checkpoint
        # ----------------------------------------------------

        last_path = CHECKPOINT_DIR / "last_checkpoint.pt"

        save_checkpoint(
            last_path,
            model,
            optimizer,
            scheduler,
            epoch,
            best_val_loss,
            best_val_f1,
            patience_counter,
            history,
        )

        # ----------------------------------------------------
        # GPU monitoring
        # ----------------------------------------------------

        if device.type == "cuda":

            memory_mb = (
                torch.cuda.max_memory_allocated(device)
                / (1024 ** 2)
            )

            print(
                f"Peak GPU memory: {memory_mb:.2f} MB"
            )

        # ----------------------------------------------------
        # Early stopping
        # ----------------------------------------------------

        if patience_counter >= patience:

            print(
                "\nEARLY STOPPING: "
                "Validation loss did not improve."
            )

            break

    # --------------------------------------------------------
    # Final report
    # --------------------------------------------------------

    report = {
        "phase": 6,
        "name": "CNN Model Training",
        "completed_epochs": len(history),
        "best_validation_loss": best_val_loss,
        "best_validation_macro_f1": best_val_f1,
        "best_checkpoint": str(
            CHECKPOINT_DIR / "best_model.pt"
        ),
        "last_checkpoint": str(
            CHECKPOINT_DIR / "last_checkpoint.pt"
        ),
        "history_file": str(HISTORY_PATH),
        "training_dataset": str(TRAIN_MANIFEST),
        "validation_dataset": str(VAL_MANIFEST),
        "test_dataset_used": False,
        "class_names": CLASS_NAMES,
        "status": "training_completed",
    }

    report_path = OUTPUT_DIR / "phase6_report.json"

    with open(report_path, "w") as file:

        json.dump(
            report,
            file,
            indent=4,
        )

    print("\n========== TRAINING COMPLETED ==========")

    print("Best validation loss:", best_val_loss)

    print("Best validation macro F1:", best_val_f1)

    print("Report saved:", report_path)


# ============================================================
# 12. COMMAND LINE
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description="Phase 6: CNN model training"
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=TrainingConfig.EPOCHS,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=TrainingConfig.BATCH_SIZE,
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=TrainingConfig.WORKERS,
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=TrainingConfig.LEARNING_RATE,
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=TrainingConfig.EARLY_STOPPING_PATIENCE,
    )

    parser.add_argument(
        "--resume",
        type=str,
        default=None,
    )

    args = parser.parse_args()

    if args.epochs < 1:
        parser.error("--epochs must be at least 1")

    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")

    if args.workers < 0:
        parser.error("--workers cannot be negative")

    if args.lr <= 0:
        parser.error("--lr must be positive")

    if args.patience < 1:
        parser.error("--patience must be at least 1")

    train_model(
        epochs=args.epochs,
        batch_size=args.batch_size,
        workers=args.workers,
        learning_rate=args.lr,
        patience=args.patience,
        resume_path=args.resume,
    )


if __name__ == "__main__":
    main()
