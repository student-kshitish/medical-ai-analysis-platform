"""
PHASE 9: MEDICAL IMAGE INFERENCE ENGINE

Loads the trained Phase 6 model and performs inference using
the Phase 4 preprocessing pipeline.

Features:
- Single X-ray image inference
- NORMAL / PNEUMONIA probabilities
- Optional Grad-CAM explanation
- JSON prediction reports
- Verification using test manifest

Research prototype only. Not for clinical diagnosis.
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

AIML_DIR = PROJECT_ROOT / "AIML-01"

CHECKPOINT_PATH = (
    AIML_DIR
    / "data/processed/phase6/checkpoints/best_model.pt"
)

NORMALIZATION_PATH = (
    AIML_DIR
    / "data/processed/phase4/normalization.json"
)

TEST_MANIFEST = (
    AIML_DIR
    / "data/processed/phase2/test_manifest.csv"
)

OUTPUT_DIR = (
    AIML_DIR
    / "data/processed/phase9"
)

EXPLANATION_DIR = (
    PROJECT_ROOT
    / "artifacts/phase9/gradcam"
)


# ============================================================
# IMPORT EXISTING PHASE MODULES
# ============================================================

sys.path.insert(0, str(AIML_DIR))

from phase4_preprocessing import MedicalImagePreprocessor
from phase5_model import ChestXRayCNN


# ============================================================
# DEVICE
# ============================================================

def get_device():
    """Select CUDA when available, otherwise CPU."""

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("GPU:", torch.cuda.get_device_name(0))
    else:
        device = torch.device("cpu")
        print("GPU unavailable. Using CPU.")

    return device


# ============================================================
# NORMALIZATION
# ============================================================

def load_normalization():
    """
    Load mean and standard deviation calculated in Phase 4.
    """

    if not NORMALIZATION_PATH.exists():
        raise FileNotFoundError(
            f"Normalization file not found: {NORMALIZATION_PATH}"
        )

    with open(NORMALIZATION_PATH, "r") as file:
        data = json.load(file)

    if "mean" not in data or "std" not in data:
        raise ValueError(
            "Normalization JSON must contain 'mean' and 'std'. "
            f"Available keys: {list(data.keys())}"
        )

    mean = float(data["mean"])
    std = float(data["std"])

    if std <= 0:
        raise ValueError("Normalization standard deviation must be positive.")

    print("Normalization mean:", mean)
    print("Normalization std:", std)

    return mean, std


# ============================================================
# INFERENCE ENGINE
# ============================================================

class MedicalInferenceEngine:

    def __init__(self, device=None):

        self.device = device or get_device()

        print("\n========== INITIALIZING PHASE 9 ==========")

        # Check required files.
        for path in [
            CHECKPOINT_PATH,
            NORMALIZATION_PATH,
        ]:
            if not path.exists():
                raise FileNotFoundError(
                    f"Required file missing: {path}"
                )

        # Load normalization.
        self.mean, self.std = load_normalization()

        # Load preprocessing.
        self.preprocessor = MedicalImagePreprocessor()

        # Load trained model.
        self.model = ChestXRayCNN().to(self.device)

        # Load checkpoint.
        checkpoint = torch.load(
            CHECKPOINT_PATH,
            map_location=self.device,
            weights_only=False,
        )

        self.model.load_state_dict(
            checkpoint["model_state_dict"],
            strict=True,
        )

        self.model.eval()

        self.epoch = int(checkpoint["epoch"])

        self.class_names = checkpoint.get(
            "class_names",
            ["NORMAL", "PNEUMONIA"],
        )

        if len(self.class_names) != 2:
            raise ValueError(
                f"Expected 2 classes, got {self.class_names}"
            )

        self.best_val_loss = checkpoint.get("best_val_loss")
        self.best_val_f1 = checkpoint.get("best_val_f1")

        print("Checkpoint loaded successfully.")
        print("Model epoch:", self.epoch)
        print("Classes:", self.class_names)
        print("Device:", self.device)
        print("PHASE 9 MODEL INITIALIZATION PASSED")

    # --------------------------------------------------------
    # IMAGE PREPROCESSING
    # --------------------------------------------------------

    def preprocess_image(self, image_path):
        """
        Apply the same Phase 4 image preprocessing and
        Phase 4 training normalization.
        """

        image_path = Path(image_path)

        if not image_path.exists():
            raise FileNotFoundError(
                f"Input image not found: {image_path}"
            )

        if not image_path.is_file():
            raise ValueError(
                f"Input path is not a file: {image_path}"
            )

        # Reuse Phase 4 preprocessing.
        image = self.preprocessor.preprocess(
            str(image_path),
            augment=False,
        )

        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image)

        if not isinstance(image, torch.Tensor):
            raise TypeError(
                "Phase 4 preprocessing must return a Tensor "
                f"or NumPy array, got {type(image)}"
            )

        image = image.float()

        # Expected shape: (1, 224, 224).
        image = image.squeeze()

        if image.ndim == 2:
            image = image.unsqueeze(0)

        if image.ndim != 3:
            raise ValueError(
                f"Unexpected preprocessed image shape: {image.shape}"
            )

        if image.shape[0] != 1:
            raise ValueError(
                "Expected one grayscale channel, "
                f"got shape {image.shape}"
            )

        if image.shape[-2:] != (224, 224):
            raise ValueError(
                "Expected 224x224 image after preprocessing, "
                f"got {image.shape[-2:]}"
            )

        # Match Phase 4 normalization.
        image = (image - self.mean) / self.std

        # Add batch dimension.
        image = image.unsqueeze(0)

        return image.to(self.device)

    # --------------------------------------------------------
    # PREDICTION
    # --------------------------------------------------------

    @torch.inference_mode()
    def predict(self, image_path):
        """
        Predict NORMAL or PNEUMONIA.
        """

        image_tensor = self.preprocess_image(image_path)

        logits = self.model(image_tensor)

        probabilities = F.softmax(
            logits,
            dim=1,
        )[0]

        predicted_index = int(
            torch.argmax(probabilities).item()
        )

        predicted_class = self.class_names[predicted_index]

        probability_dict = {
            class_name: float(probabilities[index].item())
            for index, class_name in enumerate(self.class_names)
        }

        result = {
            "image": str(Path(image_path).resolve()),
            "prediction": predicted_class,
            "predicted_class_index": predicted_index,
            "probabilities": probability_dict,
            "model_epoch": self.epoch,
            "best_validation_loss": self.best_val_loss,
            "best_validation_macro_f1": self.best_val_f1,
            "inference_time_utc": datetime.now(
                timezone.utc
            ).isoformat(),
            "status": "research_prototype",
            "clinical_warning": (
                "This output is an experimental model prediction, "
                "not a medical diagnosis. A qualified doctor must "
                "review the medical image."
            ),
        }

        return result, image_tensor

    # --------------------------------------------------------
    # GRAD-CAM
    # --------------------------------------------------------

    def generate_gradcam(self, image_tensor, predicted_index):
        """
        Generate a Grad-CAM explanation for the predicted class.

        The heatmap shows image regions that influenced the
        model's selected class. It does not establish disease
        localization or clinical reasoning.
        """

        self.model.eval()

        activation = {}
        gradient = {}

        target_layer = self.model.block4.block[5]

        def forward_hook(module, inputs, output):
            activation["value"] = output
            output.register_hook(
                lambda grad: gradient.update(
                    {"value": grad}
                )
            )

        handle = target_layer.register_forward_hook(
            forward_hook
        )

        try:
            self.model.zero_grad(set_to_none=True)

            input_tensor = image_tensor.detach().clone()
            input_tensor.requires_grad_(True)

            logits = self.model(input_tensor)

            score = logits[0, predicted_index]

            score.backward()

            features = activation["value"]
            gradients = gradient["value"]

            weights = gradients.mean(
                dim=(2, 3),
                keepdim=True,
            )

            cam = torch.sum(
                weights * features,
                dim=1,
            )

            cam = F.relu(cam)

            cam = F.interpolate(
                cam.unsqueeze(1),
                size=(224, 224),
                mode="bilinear",
                align_corners=False,
            )

            cam = cam[0, 0]

            cam = cam.detach().cpu().numpy()

            cam_min = float(cam.min())
            cam_max = float(cam.max())

            cam = (cam - cam_min) / (
                cam_max - cam_min + 1e-8
            )

            return cam

        finally:
            handle.remove()
            self.model.zero_grad(set_to_none=True)

    # --------------------------------------------------------
    # VISUALIZATION
    # --------------------------------------------------------

    def save_gradcam(self, image_tensor, cam, output_name):
        """
        Save heatmap and overlay images.
        """

        EXPLANATION_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        # Convert normalized input back to grayscale [0, 255].
        original = (
            image_tensor[0, 0].detach().cpu().numpy()
            * self.std
            + self.mean
        )

        original = np.clip(
            original,
            0.0,
            1.0,
        )

        original = (
            original * 255
        ).astype(np.uint8)

        cam_uint8 = (
            np.clip(cam, 0.0, 1.0) * 255
        ).astype(np.uint8)

        # Simple RGB heatmap.
        red = cam_uint8
        green = np.clip(
            255 - np.abs(cam_uint8.astype(np.int16) - 128) * 2,
            0,
            255,
        ).astype(np.uint8)
        blue = 255 - cam_uint8

        heatmap = np.stack(
            [red, green, blue],
            axis=-1,
        )

        original_rgb = np.repeat(
            original[..., None],
            3,
            axis=2,
        )

        overlay = (
            0.60 * original_rgb.astype(np.float32)
            + 0.40 * heatmap.astype(np.float32)
        )

        overlay = np.clip(
            overlay,
            0,
            255,
        ).astype(np.uint8)

        heatmap_path = (
            EXPLANATION_DIR
            / f"{output_name}_heatmap.png"
        )

        overlay_path = (
            EXPLANATION_DIR
            / f"{output_name}_overlay.png"
        )

        Image.fromarray(heatmap).save(heatmap_path)
        Image.fromarray(overlay).save(overlay_path)

        print("Grad-CAM heatmap:", heatmap_path)
        print("Grad-CAM overlay:", overlay_path)

        return {
            "heatmap": str(heatmap_path),
            "overlay": str(overlay_path),
        }

    # --------------------------------------------------------
    # COMPLETE INFERENCE
    # --------------------------------------------------------

    def run(self, image_path, explain=False):
        """
        Execute prediction and optionally generate Grad-CAM.
        """

        print("\n========== RUNNING INFERENCE ==========")

        result, image_tensor = self.predict(image_path)

        if explain:
            predicted_index = result[
                "predicted_class_index"
            ]

            cam = self.generate_gradcam(
                image_tensor,
                predicted_index,
            )

            output_name = (
                Path(image_path).stem
                + "_"
                + datetime.now().strftime("%Y%m%d_%H%M%S")
            )

            explanation_files = self.save_gradcam(
                image_tensor,
                cam,
                output_name,
            )

            result["explanation_generated"] = True
            result["explanation_files"] = explanation_files

        else:
            result["explanation_generated"] = False

        return result


# ============================================================
# SAVE JSON REPORT
# ============================================================

def save_result(result):

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = OUTPUT_DIR / "latest_prediction.json"

    with open(output_path, "w") as file:
        json.dump(
            result,
            file,
            indent=4,
        )

    print("\nPrediction report saved:", output_path)

    return output_path


# ============================================================
# VERIFICATION
# ============================================================

def verify_engine(engine):
    """
    Verify inference on one NORMAL and one PNEUMONIA
    image from the test manifest.

    This is an integration check, not a new evaluation.
    """

    print("\n========== PHASE 9 VERIFICATION ==========")

    if not TEST_MANIFEST.exists():
        raise FileNotFoundError(
            f"Test manifest missing: {TEST_MANIFEST}"
        )

    manifest = pd.read_csv(TEST_MANIFEST)

    required_columns = {
        "image_path",
        "label",
    }

    if not required_columns.issubset(manifest.columns):
        raise ValueError(
            f"Manifest requires columns {required_columns}"
        )

    for class_name in engine.class_names:

        matching = manifest[
            manifest["label"].astype(str).str.upper()
            == class_name.upper()
        ]

        if matching.empty:
            raise ValueError(
                f"No test images found for {class_name}"
            )

        row = matching.iloc[0]

        image_path = Path(str(row["image_path"]))

        if not image_path.is_absolute():
            image_path = PROJECT_ROOT / image_path

        print(f"\nTesting actual class: {class_name}")
        print("Image:", image_path)

        result = engine.run(
            image_path,
            explain=False,
        )

        print("Predicted:", result["prediction"])
        print("Probabilities:", result["probabilities"])

        print(
            "Correct:",
            result["prediction"].upper()
            == class_name.upper(),
        )

    print("\nPHASE 9 INFERENCE VERIFICATION COMPLETED.")


# ============================================================
# COMMAND LINE INTERFACE
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description="Phase 9 Medical Image Inference Engine"
    )

    parser.add_argument(
        "--image",
        type=str,
        help="Path to an X-ray image",
    )

    parser.add_argument(
        "--explain",
        action="store_true",
        help="Generate Grad-CAM explanation",
    )

    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify inference on one image per class",
    )

    args = parser.parse_args()

    if not args.image and not args.verify:
        parser.error(
            "Provide --image PATH or use --verify"
        )

    engine = MedicalInferenceEngine()

    if args.verify:
        verify_engine(engine)
        return

    result = engine.run(
        args.image,
        explain=args.explain,
    )

    print("\n========== PREDICTION RESULT ==========")

    print(
        json.dumps(
            result,
            indent=4,
        )
    )

    save_result(result)


if __name__ == "__main__":
    main()
