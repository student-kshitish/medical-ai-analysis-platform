"""
PHASE 4: MEDICAL IMAGE PREPROCESSING ENGINE

Supports:
- JPEG / PNG loading
- DICOM loading
- DICOM pixel extraction
- CT windowing
- Grayscale conversion
- Intensity normalization
- Aspect-ratio preserving resize
- Padding
- Training-only augmentation
- Training-set normalization statistics
- PyTorch Dataset
- PyTorch DataLoader
- Preview generation
- Pipeline verification

Initial task:
Pediatric chest X-ray classification.

Input:
224 x 224 grayscale images.

Labels:
NORMAL = 0
PNEUMONIA = 1
"""

import argparse
import csv
import json
import random

from pathlib import Path
from datetime import datetime, timezone

import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader

from PIL import (
    Image,
    ImageOps,
    ImageEnhance,
    ImageDraw,
)

import pydicom


# ============================================================
# 1. CONFIGURATION
# ============================================================

PROJECT_ROOT = Path.cwd()

MANIFEST_ROOT = (
    PROJECT_ROOT / "AIML-01/data/processed/phase2"
)

OUTPUT_ROOT = (
    PROJECT_ROOT / "AIML-01/data/processed/phase4"
)

PREVIEW_ROOT = (
    PROJECT_ROOT / "artifacts/preprocessing"
)

IMAGE_SIZE = 224

LABEL_MAP = {
    "NORMAL": 0,
    "PNEUMONIA": 1,
}

SPLITS = [
    "train",
    "val",
    "test",
]

SUPPORTED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".dcm",
}

DEFAULT_MEAN = 0.5
DEFAULT_STD = 0.5

RANDOM_SEED = 42


# ============================================================
# 2. IMAGE PREPROCESSING ENGINE
# ============================================================

class MedicalImagePreprocessor:

    def __init__(
        self,
        image_size=IMAGE_SIZE,
    ):

        self.image_size = image_size

    # --------------------------------------------------------
    # LOAD JPEG / PNG
    # --------------------------------------------------------

    def load_standard_image(self, path):

        with Image.open(path) as image:

            image = ImageOps.exif_transpose(image)

            image = image.convert("L")

            return image.copy()

    # --------------------------------------------------------
    # DICOM CT WINDOWING
    # --------------------------------------------------------

    @staticmethod
    def ct_windowing(
        pixels,
        center,
        width,
        slope=1.0,
        intercept=0.0,
    ):

        pixels = np.asarray(
            pixels,
            dtype=np.float32,
        )

        pixels = (
            pixels * float(slope)
            + float(intercept)
        )

        center = float(center)
        width = float(width)

        if width <= 0:
            raise ValueError(
                "CT window width must be positive"
            )

        lower = center - width / 2
        upper = center + width / 2

        pixels = np.clip(
            pixels,
            lower,
            upper,
        )

        pixels = (
            (pixels - lower)
            / (upper - lower)
            * 255.0
        )

        return pixels.astype(np.uint8)

    # --------------------------------------------------------
    # ROBUST INTENSITY NORMALIZATION
    # --------------------------------------------------------

    @staticmethod
    def intensity_to_uint8(pixels):

        pixels = np.asarray(
            pixels,
            dtype=np.float32,
        )

        if not np.isfinite(pixels).all():

            raise ValueError(
                "Image contains NaN or infinite pixel values"
            )

        low, high = np.percentile(
            pixels,
            [0.5, 99.5],
        )

        if high <= low:

            raise ValueError(
                "Image has no usable intensity variation"
            )

        pixels = np.clip(
            pixels,
            low,
            high,
        )

        pixels = (
            (pixels - low)
            / (high - low)
            * 255.0
        )

        return pixels.astype(np.uint8)

    # --------------------------------------------------------
    # LOAD DICOM
    # --------------------------------------------------------

    def load_dicom(self, path):

        dataset = pydicom.dcmread(
            str(path)
        )

        if "PixelData" not in dataset:

            raise ValueError(
                "DICOM does not contain PixelData"
            )

        pixels = dataset.pixel_array

        if pixels.ndim == 3:

            if pixels.shape[-1] in (3, 4):

                image = Image.fromarray(
                    pixels.astype(np.uint8)
                ).convert("L")

                return image

            raise ValueError(
                "Multi-frame or volumetric DICOM "
                "must be handled by a dedicated series pipeline"
            )

        if pixels.ndim != 2:

            raise ValueError(
                f"Unsupported DICOM pixel shape: {pixels.shape}"
            )

        modality = str(
            getattr(dataset, "Modality", "")
        ).upper()

        slope = getattr(
            dataset,
            "RescaleSlope",
            1.0,
        )

        intercept = getattr(
            dataset,
            "RescaleIntercept",
            0.0,
        )

        if modality == "CT":

            center = getattr(
                dataset,
                "WindowCenter",
                None,
            )

            width = getattr(
                dataset,
                "WindowWidth",
                None,
            )

            if center is None or width is None:

                raise ValueError(
                    "CT DICOM requires WindowCenter and WindowWidth"
                )

            if isinstance(center, pydicom.multival.MultiValue):
                center = center[0]

            if isinstance(width, pydicom.multival.MultiValue):
                width = width[0]

            pixels = self.ct_windowing(
                pixels,
                center,
                width,
                slope,
                intercept,
            )

        else:

            pixels = (
                pixels.astype(np.float32)
                * float(slope)
                + float(intercept)
            )

            pixels = self.intensity_to_uint8(
                pixels
            )

        photometric = str(
            getattr(
                dataset,
                "PhotometricInterpretation",
                "MONOCHROME2",
            )
        )

        if photometric == "MONOCHROME1":

            pixels = 255 - pixels

        image = Image.fromarray(
            pixels,
            mode="L",
        )

        return image

    # --------------------------------------------------------
    # UNIVERSAL IMAGE LOADER
    # --------------------------------------------------------

    def load_image(self, path):

        path = Path(path)

        if not path.exists():

            raise FileNotFoundError(path)

        extension = path.suffix.lower()

        if extension == ".dcm":

            return self.load_dicom(path)

        if extension in {
            ".jpg",
            ".jpeg",
            ".png",
        }:

            return self.load_standard_image(path)

        raise ValueError(
            f"Unsupported image format: {extension}"
        )

    # --------------------------------------------------------
    # ASPECT RATIO PRESERVING RESIZE
    # --------------------------------------------------------

    def resize_and_pad(self, image):

        image = image.convert("L")

        original_width, original_height = image.size

        if original_width <= 0 or original_height <= 0:

            raise ValueError(
                "Invalid image dimensions"
            )

        image.thumbnail(
            (
                self.image_size,
                self.image_size,
            ),
            Image.Resampling.LANCZOS,
        )

        canvas = Image.new(
            "L",
            (
                self.image_size,
                self.image_size,
            ),
            color=0,
        )

        x = (
            self.image_size - image.width
        ) // 2

        y = (
            self.image_size - image.height
        ) // 2

        canvas.paste(
            image,
            (x, y),
        )

        return canvas

    # --------------------------------------------------------
    # TRAINING AUGMENTATION
    # --------------------------------------------------------

    @staticmethod
    def augment(image):

        # Small rotation.
        angle = random.uniform(
            -7.0,
            7.0,
        )

        image = image.rotate(
            angle,
            resample=Image.Resampling.BILINEAR,
            fillcolor=0,
        )

        # Mild contrast variation.
        contrast = random.uniform(
            0.90,
            1.10,
        )

        image = ImageEnhance.Contrast(
            image
        ).enhance(contrast)

        # Mild brightness variation.
        brightness = random.uniform(
            0.90,
            1.10,
        )

        image = ImageEnhance.Brightness(
            image
        ).enhance(brightness)

        return image

    # --------------------------------------------------------
    # COMPLETE IMAGE PIPELINE
    # --------------------------------------------------------

    def preprocess(
        self,
        path,
        augment=False,
    ):

        image = self.load_image(path)

        image = self.resize_and_pad(
            image
        )

        if augment:

            image = self.augment(
                image
            )

        pixels = np.asarray(
            image,
            dtype=np.float32,
        )

        pixels = pixels / 255.0

        # Shape: 1 x H x W
        tensor = torch.from_numpy(
            pixels.copy()
        ).unsqueeze(0)

        return tensor


# ============================================================
# 3. DATASET MANAGER
# ============================================================

class MedicalImageDataset(Dataset):

    def __init__(
        self,
        manifest_path,
        preprocessor,
        mean=DEFAULT_MEAN,
        std=DEFAULT_STD,
        augment=False,
    ):

        self.manifest_path = Path(
            manifest_path
        )

        self.preprocessor = preprocessor

        self.mean = float(mean)
        self.std = float(std)

        self.augment = augment

        if self.std <= 0:

            raise ValueError(
                "Normalization standard deviation must be positive"
            )

        self.records = []

        if not self.manifest_path.exists():

            raise FileNotFoundError(
                self.manifest_path
            )

        with open(
            self.manifest_path,
            newline="",
            encoding="utf-8",
        ) as file:

            reader = csv.DictReader(file)

            for row in reader:

                image_path = Path(
                    row["image_path"]
                )

                if not image_path.is_absolute():

                    image_path = (
                        PROJECT_ROOT / image_path
                    )

                if not image_path.exists():

                    raise FileNotFoundError(
                        image_path
                    )

                label = row["label"].upper()

                if label not in LABEL_MAP:

                    raise ValueError(
                        f"Unknown label: {label}"
                    )

                if image_path.suffix.lower() not in SUPPORTED_EXTENSIONS:

                    raise ValueError(
                        f"Unsupported image: {image_path}"
                    )

                self.records.append({
                    "path": image_path,
                    "label": LABEL_MAP[label],
                })

        if not self.records:

            raise RuntimeError(
                f"No images in manifest: {manifest_path}"
            )

    def __len__(self):

        return len(self.records)

    def __getitem__(self, index):

        record = self.records[index]

        image = self.preprocessor.preprocess(
            record["path"],
            augment=self.augment,
        )

        # Training-set statistics are applied to all splits.
        image = (
            image - self.mean
        ) / self.std

        label = torch.tensor(
            record["label"],
            dtype=torch.long,
        )

        return {
            "image": image,
            "label": label,
            "path": str(record["path"]),
        }


# ============================================================
# 4. TRAINING DATASET STATISTICS
# ============================================================

class DatasetStatistics:

    def __init__(self, preprocessor):

        self.preprocessor = preprocessor

    def calculate(self, manifest_path):

        dataset = MedicalImageDataset(
            manifest_path=manifest_path,
            preprocessor=self.preprocessor,
            augment=False,
        )

        print("\n========== COMPUTING TRAINING STATISTICS ==========")

        total_pixels = 0

        pixel_sum = 0.0

        pixel_square_sum = 0.0

        for index in range(len(dataset)):

            record = dataset.records[index]

            image = self.preprocessor.preprocess(
                record["path"],
                augment=False,
            )

            pixels = image.numpy().astype(
                np.float64
            )

            total_pixels += pixels.size

            pixel_sum += pixels.sum()

            pixel_square_sum += np.square(
                pixels
            ).sum()

            if (index + 1) % 500 == 0:

                print(
                    f"Processed {index + 1}/{len(dataset)}"
                )

        mean = pixel_sum / total_pixels

        variance = (
            pixel_square_sum / total_pixels
            - mean ** 2
        )

        std = np.sqrt(
            max(variance, 0.0)
        )

        if std <= 0:

            raise ValueError(
                "Calculated standard deviation is zero"
            )

        result = {
            "mean": float(mean),
            "std": float(std),
            "total_images": len(dataset),
            "total_pixels": total_pixels,
            "source_split": "train",
            "image_size": IMAGE_SIZE,
            "calculated_at": datetime.now(
                timezone.utc
            ).isoformat(),
        }

        OUTPUT_ROOT.mkdir(
            parents=True,
            exist_ok=True,
        )

        output_path = (
            OUTPUT_ROOT / "normalization.json"
        )

        with open(
            output_path,
            "w",
        ) as file:

            json.dump(
                result,
                file,
                indent=4,
            )

        print("\nMean:", mean)

        print("Standard deviation:", std)

        print("Saved:", output_path)

        return result


# ============================================================
# 5. DATASET PIPELINE MANAGER
# ============================================================

class Phase4Pipeline:

    def __init__(self):

        random.seed(RANDOM_SEED)

        np.random.seed(RANDOM_SEED)

        torch.manual_seed(RANDOM_SEED)

        self.preprocessor = MedicalImagePreprocessor()

        OUTPUT_ROOT.mkdir(
            parents=True,
            exist_ok=True,
        )

        PREVIEW_ROOT.mkdir(
            parents=True,
            exist_ok=True,
        )

    # --------------------------------------------------------
    # LOAD NORMALIZATION PARAMETERS
    # --------------------------------------------------------

    def load_normalization(self):

        path = (
            OUTPUT_ROOT / "normalization.json"
        )

        if not path.exists():

            raise FileNotFoundError(
                "Normalization statistics not found. "
                "Run the stats command first."
            )

        with open(path) as file:

            return json.load(file)

    # --------------------------------------------------------
    # COMPUTE TRAINING STATISTICS
    # --------------------------------------------------------

    def calculate_statistics(self):

        manifest = (
            MANIFEST_ROOT / "train_manifest.csv"
        )

        analyzer = DatasetStatistics(
            self.preprocessor
        )

        return analyzer.calculate(
            manifest
        )

    # --------------------------------------------------------
    # CREATE DATASET
    # --------------------------------------------------------

    def create_dataset(
        self,
        split,
        augment=False,
    ):

        if split not in SPLITS:

            raise ValueError(
                f"Invalid split: {split}"
            )

        normalization = self.load_normalization()

        manifest = (
            MANIFEST_ROOT / f"{split}_manifest.csv"
        )

        dataset = MedicalImageDataset(
            manifest_path=manifest,
            preprocessor=self.preprocessor,
            mean=normalization["mean"],
            std=normalization["std"],
            augment=augment,
        )

        return dataset

    # --------------------------------------------------------
    # CREATE DATALOADER
    # --------------------------------------------------------

    def create_dataloader(
        self,
        split,
        batch_size=16,
        workers=2,
    ):

        dataset = self.create_dataset(
            split=split,
            augment=(split == "train"),
        )

        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=False,
        )

        return loader

    # --------------------------------------------------------
    # VERIFY BATCH
    # --------------------------------------------------------

    def verify_batch(
        self,
        split="train",
        batch_size=16,
        workers=2,
    ):

        print(
            f"\n========== VERIFYING {split.upper()} =========="
        )

        loader = self.create_dataloader(
            split,
            batch_size,
            workers,
        )

        batch = next(iter(loader))

        images = batch["image"]

        labels = batch["label"]

        print("Image shape:", tuple(images.shape))

        print("Label shape:", tuple(labels.shape))

        print("Image dtype:", images.dtype)

        print("Label dtype:", labels.dtype)

        print("Minimum:", images.min().item())

        print("Maximum:", images.max().item())

        print("Mean:", images.mean().item())

        print("Standard deviation:", images.std().item())

        print("Labels:", labels.tolist())

        print("Paths:", batch["path"][:3])

        assert images.ndim == 4

        assert images.shape[1] == 1

        assert images.shape[2] == IMAGE_SIZE

        assert images.shape[3] == IMAGE_SIZE

        assert labels.ndim == 1

        assert torch.isfinite(images).all()

        assert torch.all(
            (labels == 0) | (labels == 1)
        )

        print("\nBatch verification PASSED.")

    # --------------------------------------------------------
    # CREATE PREVIEW
    # --------------------------------------------------------

    def create_preview(
        self,
        split="train",
        count=8,
    ):

        dataset = self.create_dataset(
            split=split,
            augment=False,
        )

        count = min(
            count,
            len(dataset),
        )

        columns = 4

        rows = (
            count + columns - 1
        ) // columns

        cell_size = IMAGE_SIZE + 30

        canvas = Image.new(
            "RGB",
            (
                columns * cell_size,
                rows * cell_size,
            ),
            color="white",
        )

        draw = ImageDraw.Draw(canvas)

        mean = dataset.mean

        std = dataset.std

        for index in range(count):

            item = dataset[index]

            tensor = item["image"]

            # Reverse normalization for visualization.
            pixels = (
                tensor.squeeze(0).numpy()
                * std + mean
            )

            pixels = np.clip(
                pixels * 255.0,
                0,
                255,
            ).astype(np.uint8)

            image = Image.fromarray(
                pixels,
                mode="L",
            ).convert("RGB")

            x = (
                index % columns
            ) * cell_size

            y = (
                index // columns
            ) * cell_size

            canvas.paste(
                image,
                (x, y),
            )

            label = (
                "NORMAL"
                if item["label"].item() == 0
                else "PNEUMONIA"
            )

            draw.text(
                (
                    x,
                    y + IMAGE_SIZE + 4,
                ),
                label,
                fill="black",
            )

        output_path = (
            PREVIEW_ROOT / f"{split}_preview.png"
        )

        canvas.save(output_path)

        print("Preview saved:", output_path)

        return output_path

    # --------------------------------------------------------
    # GENERATE PHASE REPORT
    # --------------------------------------------------------

    def generate_report(self):

        normalization = self.load_normalization()

        report = {
            "phase": 4,
            "name": "Medical Image Preprocessing",
            "created_at": datetime.now(
                timezone.utc
            ).isoformat(),
            "image_size": IMAGE_SIZE,
            "channels": 1,
            "labels": LABEL_MAP,
            "normalization": normalization,
            "augmentation": {
                "training_only": True,
                "rotation_degrees": 7,
                "brightness_range": [0.90, 1.10],
                "contrast_range": [0.90, 1.10],
                "horizontal_flip": False,
                "vertical_flip": False,
            },
            "formats": [
                "JPEG",
                "PNG",
                "DICOM",
            ],
            "status": "verified",
        }

        output_path = (
            OUTPUT_ROOT / "phase4_report.json"
        )

        with open(
            output_path,
            "w",
        ) as file:

            json.dump(
                report,
                file,
                indent=4,
            )

        print("Phase 4 report saved:", output_path)


# ============================================================
# 6. COMMAND LINE INTERFACE
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description="Phase 4 Medical Image Preprocessing"
    )

    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
    )

    # Statistics
    subparsers.add_parser(
        "stats",
        help="Calculate training normalization statistics",
    )

    # Batch verification
    check_parser = subparsers.add_parser(
        "check",
        help="Verify preprocessing and DataLoader",
    )

    check_parser.add_argument(
        "--split",
        choices=SPLITS,
        default="train",
    )

    check_parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
    )

    check_parser.add_argument(
        "--workers",
        type=int,
        default=2,
    )

    # Preview
    preview_parser = subparsers.add_parser(
        "preview",
        help="Generate processed image preview",
    )

    preview_parser.add_argument(
        "--split",
        choices=SPLITS,
        default="train",
    )

    preview_parser.add_argument(
        "--count",
        type=int,
        default=8,
    )

    # Report
    subparsers.add_parser(
        "report",
        help="Generate Phase 4 report",
    )

    args = parser.parse_args()

    pipeline = Phase4Pipeline()

    if args.command == "stats":

        pipeline.calculate_statistics()

    elif args.command == "check":

        pipeline.verify_batch(
            split=args.split,
            batch_size=args.batch_size,
            workers=args.workers,
        )

    elif args.command == "preview":

        pipeline.create_preview(
            split=args.split,
            count=args.count,
        )

    elif args.command == "report":

        pipeline.generate_report()


if __name__ == "__main__":

    main()
