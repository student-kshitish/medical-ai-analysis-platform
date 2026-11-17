import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path.cwd()

DATASET_PATH = (
    PROJECT_ROOT
    / "AIML-01/data/raw/pediatric_chest_xray/extracted"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "AIML-01/data/processed/phase2"
)

CLASSES = ["NORMAL", "PNEUMONIA"]

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

VALIDATION_SIZE = 0.20
RANDOM_STATE = 42


# ============================================================
# PHASE 2 DATASET PIPELINE
# ============================================================

class Phase2DatasetPipeline:

    def __init__(self):

        self.records = []

        self.duplicate_groups = []

        self.invalid_images = []

        self.filename_audit = {}

        self.final_manifests = {}

        OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    # 1. DISCOVER AND VALIDATE IMAGES
    # --------------------------------------------------------

    def discover_images(self):

        print("\n[1/9] Discovering dataset images...")

        for split in ["train", "val", "test"]:

            for label in CLASSES:

                folder = DATASET_PATH / split / label

                if not folder.exists():
                    raise FileNotFoundError(folder)

                for path in sorted(folder.iterdir()):

                    if not path.is_file():
                        continue

                    if path.suffix.lower() not in IMAGE_EXTENSIONS:
                        continue

                    try:

                        with Image.open(path) as image:

                            image.verify()

                        with Image.open(path) as image:

                            width, height = image.size

                            if width < 32 or height < 32:
                                raise ValueError(
                                    "Image dimensions too small"
                                )

                        sha256 = self.calculate_hash(path)

                        record = {
                            "path": path,
                            "original_split": split,
                            "label": label,
                            "sha256": sha256,
                            "width": width,
                            "height": height,
                        }

                        self.records.append(record)

                    except Exception as error:

                        self.invalid_images.append({
                            "path": str(path),
                            "error": str(error),
                        })

        print("Discovered images:", len(self.records))
        print("Invalid images:", len(self.invalid_images))

        if self.invalid_images:
            raise RuntimeError(
                "Invalid images detected. Resolve them before proceeding."
            )

        if not self.records:
            raise RuntimeError("No images found.")

    # --------------------------------------------------------
    # 2. SHA256 HASHING
    # --------------------------------------------------------

    @staticmethod
    def calculate_hash(path):

        digest = hashlib.sha256()

        with open(path, "rb") as file:

            for chunk in iter(
                lambda: file.read(1024 * 1024),
                b""
            ):

                digest.update(chunk)

        return digest.hexdigest()

    # --------------------------------------------------------
    # 3. FILENAME AND PATIENT ID AUDIT
    # --------------------------------------------------------

    def audit_filenames(self):

        print("\n[2/9] Auditing filenames...")

        names = [
            record["path"].name
            for record in self.records
        ]

        patterns = {
            "person": "person",
            "patient": "patient",
            "subject": "subject",
        }

        results = {}

        for identifier, keyword in patterns.items():

            matches = [
                name for name in names
                if keyword in name.lower()
            ]

            results[identifier] = len(matches)

        self.filename_audit = {
            "total_filenames": len(names),
            "identifier_matches": results,
            "patient_level_split_available": False,
            "split_method": "image_level_stratified",
        }

        print("Filename examples:", names[:10])
        print("Identifier matches:", results)

        print(
            "Patient-level splitting cannot be established "
            "from the available filenames."
        )

    # --------------------------------------------------------
    # 4. DUPLICATE DETECTION
    # --------------------------------------------------------

    def detect_duplicates(self):

        print("\n[3/9] Detecting exact duplicate images...")

        hash_groups = defaultdict(list)

        for record in self.records:

            hash_groups[record["sha256"]].append(record)

        for sha256, group in hash_groups.items():

            if len(group) > 1:

                labels = {
                    record["label"]
                    for record in group
                }

                if len(labels) > 1:

                    raise RuntimeError(
                        "Identical image content has conflicting labels: "
                        + sha256
                    )

                self.duplicate_groups.append({
                    "sha256": sha256,
                    "count": len(group),
                    "files": [
                        str(record["path"])
                        for record in group
                    ],
                    "original_splits": [
                        record["original_split"]
                        for record in group
                    ],
                })

        cross_split = [
            group
            for group in self.duplicate_groups
            if len(set(group["original_splits"])) > 1
        ]

        print("Duplicate groups:", len(self.duplicate_groups))
        print("Cross-split duplicate groups:", len(cross_split))

        if cross_split:

            raise RuntimeError(
                "Cross-split duplicates detected. "
                "Review before preparing the final dataset."
            )

    # --------------------------------------------------------
    # 5. DUPLICATE HANDLING
    # --------------------------------------------------------

    @staticmethod
    def deduplicate(records):

        seen = set()

        unique_records = []

        removed = 0

        for record in sorted(
            records,
            key=lambda item: str(item["path"])
        ):

            sha256 = record["sha256"]

            if sha256 in seen:

                removed += 1
                continue

            seen.add(sha256)

            unique_records.append(record)

        return unique_records, removed

    # --------------------------------------------------------
    # 6. TRAIN / VALIDATION / TEST SPLITTING
    # --------------------------------------------------------

    def prepare_splits(self):

        print("\n[4/9] Preparing final dataset splits...")

        training_pool = [
            record
            for record in self.records
            if record["original_split"] in ["train", "val"]
        ]

        original_test = [
            record
            for record in self.records
            if record["original_split"] == "test"
        ]

        training_pool, train_removed = self.deduplicate(
            training_pool
        )

        original_test, test_removed = self.deduplicate(
            original_test
        )

        print("Training pool after deduplication:", len(training_pool))
        print("Test set after deduplication:", len(original_test))

        labels = [
            record["label"]
            for record in training_pool
        ]

        train_records, val_records = train_test_split(
            training_pool,
            test_size=VALIDATION_SIZE,
            random_state=RANDOM_STATE,
            stratify=labels,
        )

        self.final_manifests = {
            "train": list(train_records),
            "val": list(val_records),
            "test": list(original_test),
        }

        print("Removed training duplicates:", train_removed)
        print("Removed test duplicates:", test_removed)

        for split, records in self.final_manifests.items():

            print(f"{split}: {len(records)} images")

    # --------------------------------------------------------
    # 7. GENERATE CSV MANIFESTS
    # --------------------------------------------------------

    def generate_manifests(self):

        print("\n[5/9] Generating dataset manifests...")

        for split, records in self.final_manifests.items():

            output_file = OUTPUT_PATH / f"{split}_manifest.csv"

            with open(
                output_file,
                "w",
                newline="",
                encoding="utf-8"
            ) as file:

                writer = csv.DictWriter(
                    file,
                    fieldnames=[
                        "image_path",
                        "label",
                        "sha256",
                        "width",
                        "height",
                        "original_split",
                        "final_split",
                    ],
                )

                writer.writeheader()

                for record in records:

                    writer.writerow({
                        "image_path": str(
                            record["path"].relative_to(PROJECT_ROOT)
                        ),
                        "label": record["label"],
                        "sha256": record["sha256"],
                        "width": record["width"],
                        "height": record["height"],
                        "original_split": record["original_split"],
                        "final_split": split,
                    })

            print("Saved:", output_file)

    # --------------------------------------------------------
    # 8. STATISTICS AND CLASS WEIGHTS
    # --------------------------------------------------------

    def calculate_statistics(self):

        print("\n[6/9] Calculating final statistics...")

        statistics = {}

        for split, records in self.final_manifests.items():

            counts = Counter(
                record["label"]
                for record in records
            )

            statistics[split] = {
                label: counts.get(label, 0)
                for label in CLASSES
            }

        train_counts = statistics["train"]

        total_train = sum(train_counts.values())

        class_weights = {}

        for label in CLASSES:

            count = train_counts[label]

            if count == 0:
                raise RuntimeError(
                    f"No training images for {label}"
                )

            class_weights[label] = (
                total_train / (len(CLASSES) * count)
            )

        return {
            "split_distribution": statistics,
            "training_class_weights": class_weights,
        }

    # --------------------------------------------------------
    # 9. FINAL INTEGRITY VERIFICATION
    # --------------------------------------------------------

    def verify_integrity(self):

        print("\n[7/9] Verifying final dataset...")

        split_hashes = {}

        for split, records in self.final_manifests.items():

            hashes = [
                record["sha256"]
                for record in records
            ]

            if len(hashes) != len(set(hashes)):

                raise RuntimeError(
                    f"Duplicates remain inside {split}"
                )

            split_hashes[split] = set(hashes)

        splits = list(split_hashes)

        for index, first in enumerate(splits):

            for second in splits[index + 1:]:

                overlap = (
                    split_hashes[first]
                    & split_hashes[second]
                )

                if overlap:

                    raise RuntimeError(
                        f"Data leakage between {first} and {second}"
                    )

        print("No exact duplicates within final splits.")
        print("No exact duplicate overlap across final splits.")

        print("Final integrity checks passed.")

    # --------------------------------------------------------
    # REPORT GENERATION
    # --------------------------------------------------------

    def save_reports(self, statistics):

        print("\n[8/9] Saving reports...")

        total = sum(
            len(records)
            for records in self.final_manifests.values()
        )

        report = {
            "dataset": "Pediatric Chest X-ray",
            "total_final_images": total,
            "original_images": len(self.records),
            "duplicate_groups": len(self.duplicate_groups),
            "invalid_images": len(self.invalid_images),
            "filename_audit": self.filename_audit,
            "statistics": statistics,
            "random_state": RANDOM_STATE,
            "validation_size": VALIDATION_SIZE,
            "patient_level_split": False,
            "duplicate_details": self.duplicate_groups,
            "split_integrity": "passed",
        }

        report_path = OUTPUT_PATH / "phase2_report.json"

        with open(report_path, "w") as file:

            json.dump(report, file, indent=4)

        print("Saved:", report_path)

    # --------------------------------------------------------
    # EXECUTION
    # --------------------------------------------------------

    def run(self):

        print("\n========================================")
        print("       PHASE 2 DATASET PIPELINE")
        print("========================================")

        self.discover_images()

        self.audit_filenames()

        self.detect_duplicates()

        self.prepare_splits()

        self.generate_manifests()

        statistics = self.calculate_statistics()

        self.verify_integrity()

        self.save_reports(statistics)

        print("\n[9/9] PHASE 2 COMPLETED SUCCESSFULLY")

        print("\nFinal class distribution:")

        for split, counts in (
            statistics["split_distribution"].items()
        ):

            print(split, counts)

        print("\nTraining class weights:")

        for label, weight in (
            statistics["training_class_weights"].items()
        ):

            print(f"{label}: {weight:.4f}")

        print("\nOutput directory:", OUTPUT_PATH)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    pipeline = Phase2DatasetPipeline()

    pipeline.run()
