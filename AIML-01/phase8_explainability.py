"""
PHASE 8: GRAD-CAM EXPLAINABILITY

Generates Grad-CAM visualizations for the trained CNN.

Uses:
    Phase 4: MedicalImageDataset
    Phase 5: ChestXRayCNN
    Phase 6: best_model.pt

Does not train or modify the model.
"""

import argparse
import csv
import json

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from PIL import Image

from phase4_preprocessing import Phase4Pipeline
from phase5_model import ChestXRayCNN


# ============================================================
# 1. PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "AIML-01/data/processed/phase6/checkpoints/best_model.pt"
)

TEST_MANIFEST = (
    PROJECT_ROOT
    / "AIML-01/data/processed/phase2/test_manifest.csv"
)

NORMALIZATION_PATH = (
    PROJECT_ROOT
    / "AIML-01/data/processed/phase4/normalization.json"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "artifacts/gradcam"
)

REPORT_PATH = (
    PROJECT_ROOT
    / "AIML-01/data/processed/phase8/phase8_report.json"
)

CLASS_NAMES = [
    "NORMAL",
    "PNEUMONIA",
]


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

        print("Using CPU")

    return device


# ============================================================
# 3. LOAD MODEL
# ============================================================

def load_model(device):

    if not CHECKPOINT_PATH.exists():

        raise FileNotFoundError(
            f"Checkpoint missing: {CHECKPOINT_PATH}"
        )

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

    if checkpoint.get("class_names") != CLASS_NAMES:

        raise ValueError(
            "Checkpoint class labels do not match."
        )

    model = ChestXRayCNN()

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.to(device)

    model.eval()

    print("Loaded best checkpoint.")

    print("Checkpoint epoch:", checkpoint["epoch"])

    return model


# ============================================================
# 4. NORMALIZATION STATISTICS
# ============================================================

def load_normalization():

    with open(NORMALIZATION_PATH) as file:

        data = json.load(file)

    mean = float(data["mean"])
    std = float(data["std"])

    if std <= 0:
        raise ValueError("Invalid normalization standard deviation.")

    return mean, std


# ============================================================
# 5. SELECT TEST IMAGES
# ============================================================

def select_test_indices(samples_per_class):

    selected = {
        "NORMAL": [],
        "PNEUMONIA": [],
    }

    with open(TEST_MANIFEST, newline="") as file:

        reader = csv.DictReader(file)

        for index, row in enumerate(reader):

            label = row["label"].strip().upper()

            if label not in selected:
                raise ValueError(f"Unexpected label: {label}")

            if len(selected[label]) < samples_per_class:

                selected[label].append(index)

            if all(
                len(indices) >= samples_per_class
                for indices in selected.values()
            ):
                break

    for label, indices in selected.items():

        if len(indices) < samples_per_class:

            raise ValueError(
                f"Not enough test images for {label}"
            )

    indices = (
        selected["NORMAL"]
        + selected["PNEUMONIA"]
    )

    print("\nSelected test images:")

    print("NORMAL:", len(selected["NORMAL"]))

    print("PNEUMONIA:", len(selected["PNEUMONIA"]))

    return indices


# ============================================================
# 6. GRAD-CAM CLASS
# ============================================================

class GradCAM:

    def __init__(self, model, target_layer):

        self.model = model
        self.activations = None
        self.gradients = None

        self.hook = target_layer.register_forward_hook(
            self._forward_hook
        )

    def _forward_hook(self, module, inputs, output):

        self.activations = output

        # Capture gradients of the target activation.
        output.register_hook(
            self._gradient_hook
        )

    def _gradient_hook(self, gradient):

        self.gradients = gradient

    def generate(self, image, class_index):

        self.model.zero_grad(set_to_none=True)

        self.activations = None
        self.gradients = None

        logits = self.model(image)

        if not 0 <= class_index < len(CLASS_NAMES):

            raise ValueError("Invalid class index.")

        score = logits[:, class_index].sum()

        score.backward()

        if self.activations is None or self.gradients is None:

            raise RuntimeError(
                "Could not capture Grad-CAM activations or gradients."
            )

        # Average gradients over spatial dimensions.
        weights = self.gradients.mean(
            dim=(2, 3),
            keepdim=True,
        )

        # Weighted sum of feature maps.
        cam = (
            weights * self.activations
        ).sum(dim=1, keepdim=True)

        # Keep positive contributions.
        cam = F.relu(cam)

        # Resize to original input resolution.
        cam = F.interpolate(
            cam,
            size=(224, 224),
            mode="bilinear",
            align_corners=False,
        )

        cam = cam[0, 0]

        # Normalize between 0 and 1.
        cam_min = cam.min()
        cam_max = cam.max()

        if (cam_max - cam_min).item() > 1e-8:

            cam = (
                cam - cam_min
            ) / (
                cam_max - cam_min
            )

        else:

            cam = torch.zeros_like(cam)

        return (
            cam.detach().cpu().numpy(),
            logits.detach(),
        )

    def close(self):

        self.hook.remove()


# ============================================================
# 7. CONVERT NORMALIZED TENSOR TO IMAGE
# ============================================================

def tensor_to_grayscale(tensor, mean, std):

    image = tensor.detach().cpu().numpy()

    # Remove batch/channel dimensions.
    image = np.squeeze(image)

    # Ensure exactly two-dimensional grayscale image.
    if image.ndim != 2:
        raise ValueError(
            f"Expected 2D grayscale image, got {image.shape}"
        )

    # Reverse Phase 4 normalization.
    image = image * std + mean

    image = np.clip(
        image,
        0.0,
        1.0,
    )

    image = (image * 255).astype(np.uint8)

    return image

# ============================================================
# 8. GENERATE COLOR HEATMAP
# ============================================================

def create_heatmap(cam):

    cam = np.clip(
        cam,
        0.0,
        1.0,
    )

    # Simple blue-to-red heatmap.
    red = np.clip(
        2.0 * cam,
        0,
        1,
    )

    green = np.clip(
        2.0 * (1.0 - np.abs(cam - 0.5)),
        0,
        1,
    )

    blue = np.clip(
        1.0 - 2.0 * cam,
        0,
        1,
    )

    heatmap = np.stack(
        [red, green, blue],
        axis=-1,
    )

    return (
        heatmap * 255
    ).astype(np.uint8)


# ============================================================
# 9. CREATE OVERLAY
# ============================================================

def create_overlay(original, cam, alpha=0.40):

    original = np.squeeze(original)
    cam = np.squeeze(cam)

    if original.ndim != 2:
        raise ValueError(
            f"Original image must be 2D, got {original.shape}"
        )

    if cam.ndim != 2:
        raise ValueError(
            f"Grad-CAM must be 2D, got {cam.shape}"
        )

    if original.shape != cam.shape:
        raise ValueError(
            f"Image and CAM dimensions differ: "
            f"{original.shape} vs {cam.shape}"
        )

    original_rgb = np.repeat(
        original[..., np.newaxis],
        3,
        axis=2,
    )

    heatmap = create_heatmap(cam)

    overlay = (
        (1 - alpha) * original_rgb.astype(np.float32)
        + alpha * heatmap.astype(np.float32)
    )

    overlay = np.clip(
        overlay,
        0,
        255,
    ).astype(np.uint8)

    if overlay.shape != (224, 224, 3):
        raise ValueError(
            f"Unexpected overlay shape: {overlay.shape}"
        )

    return overlay

# ============================================================
# 10. PROCESS ONE IMAGE
# ============================================================

def explain_image(
    model,
    gradcam,
    sample,
    index,
    mean,
    std,
    device,
):

    image = sample["image"].unsqueeze(0).to(device)

    actual_label = int(sample["label"])

    image_path = str(sample["path"])

    # Grad-CAM requires gradients even though this is inference.
    with torch.enable_grad():

        cam, logits = gradcam.generate(
            image,
            class_index=0,
        )

        probabilities = torch.softmax(
            logits,
            dim=1,
        )

        predicted_class = int(
            probabilities.argmax(dim=1).item()
        )

        pneumonia_probability = float(
            probabilities[0, 1].item()
        )

        # Generate CAM for the model's predicted class.
        cam, _ = gradcam.generate(
            image,
            class_index=predicted_class,
        )

    original = tensor_to_grayscale(
        sample["image"],
        mean,
        std,
    )

    overlay = create_overlay(
        original,
        cam,
    )
    print("Original image shape:", original.shape)
    print("Grad-CAM shape:", cam.shape)
    print("Overlay shape:", overlay.shape)

    assert original.shape == (224, 224)
    assert cam.shape == (224, 224)
    assert overlay.shape == (224, 224, 3)
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    filename = f"gradcam_{index:03d}.png"

    output_path = OUTPUT_DIR / filename

    # Save overlay.
    Image.fromarray(overlay).save(
        output_path
    )

    # Save standalone heatmap.
    heatmap_path = (
        OUTPUT_DIR
        / f"heatmap_{index:03d}.png"
    )

    Image.fromarray(
        create_heatmap(cam)
    ).save(heatmap_path)

    result = {

        "index": index,

        "image_path": image_path,

        "actual_label": CLASS_NAMES[actual_label],

        "predicted_label": CLASS_NAMES[predicted_class],

        "pneumonia_probability": pneumonia_probability,

        "correct": (
            actual_label == predicted_class
        ),

        "explained_class": CLASS_NAMES[predicted_class],

        "gradcam_overlay": str(output_path),

        "heatmap": str(heatmap_path),
    }

    print(
        f"\nImage {index}: "
        f"Actual={result['actual_label']}, "
        f"Predicted={result['predicted_label']}, "
        f"P(Pneumonia)={pneumonia_probability:.4f}"
    )

    print("Saved:", output_path)

    return result


# ============================================================
# 11. SAVE REPORT
# ============================================================

def save_report(results):

    REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    report = {

        "phase": 8,

        "name": "Grad-CAM Explainability",

        "created_at": datetime.now(
            timezone.utc
        ).isoformat(),

        "checkpoint": str(CHECKPOINT_PATH),

        "target_layer": "block4.block[5]",

        "method": "Gradient-weighted Class Activation Mapping",

        "explained_class": "Model-predicted class",

        "samples_explained": len(results),

        "results": results,

        "limitations": [
            "Grad-CAM visualizations are not proof of disease localization.",
            "Highlighted regions do not establish medical causality.",
            "Heatmaps depend on the model and selected convolutional layer.",
            "A clinician must interpret medical images independently.",
        ],

        "status": "explainability_completed",
    }

    with open(REPORT_PATH, "w") as file:

        json.dump(
            report,
            file,
            indent=4,
            allow_nan=False,
        )

    print("\nReport saved:", REPORT_PATH)


# ============================================================
# 12. MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description="Phase 8: Grad-CAM Explainability"
    )

    parser.add_argument(
        "--samples-per-class",
        type=int,
        default=4,
    )

    args = parser.parse_args()

    if args.samples_per_class < 1:
        parser.error("--samples-per-class must be positive")

    device = get_device()

    model = load_model(device)

    mean, std = load_normalization()

    pipeline = Phase4Pipeline()

    dataset = pipeline.create_dataset(
        split="test",
        augment=False,
    )

    indices = select_test_indices(
        args.samples_per_class
    )

    # Final convolutional block's last ReLU.
    target_layer = model.block4.block[5]

    gradcam = GradCAM(
        model,
        target_layer,
    )

    results = []

    try:

        print("\n========== GENERATING GRAD-CAM ==========")

        for output_index, dataset_index in enumerate(indices, start=1):

            sample = dataset[dataset_index]

            result = explain_image(
                model=model,
                gradcam=gradcam,
                sample=sample,
                index=output_index,
                mean=mean,
                std=std,
                device=device,
            )

            results.append(result)

    finally:

        gradcam.close()

    save_report(results)

    print("\nPHASE 8 COMPLETED.")


if __name__ == "__main__":
    main()
