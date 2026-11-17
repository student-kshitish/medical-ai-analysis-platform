"""
Phase 3: DICOM Processing and Orthanc Integration

Capabilities:
- DICOM discovery and validation
- Metadata extraction
- Pixel-data decoding
- CT windowing
- DICOM preview export
- Synthetic DICOM generation from JPEG/PNG
- Orthanc connectivity
- DICOM upload, listing and retrieval

Important:
Synthetic DICOM files are for software testing only.
They are not original clinical DICOM studies.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pydicom
import requests

from PIL import Image

from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import (
    ExplicitVRLittleEndian,
    SecondaryCaptureImageStorage,
    generate_uid,
)


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path.cwd()

DICOM_ROOT = (
    PROJECT_ROOT / "AIML-01/data/dicom"
)

SYNTHETIC_ROOT = DICOM_ROOT / "synthetic"

PREVIEW_ROOT = (
    PROJECT_ROOT / "artifacts/dicom_previews"
)

REPORT_ROOT = (
    PROJECT_ROOT / "AIML-01/data/metadata"
)

ORTHANC_URL = os.getenv(
    "ORTHANC_URL",
    "http://localhost:8042"
).rstrip("/")

ORTHANC_USERNAME = os.getenv(
    "ORTHANC_USERNAME",
    ""
)

ORTHANC_PASSWORD = os.getenv(
    "ORTHANC_PASSWORD",
    ""
)

REQUEST_TIMEOUT = 30

SUPPORTED_IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png"
}


# ============================================================
# MAIN DICOM PIPELINE
# ============================================================

class DICOMPipeline:

    def __init__(self):

        self.dicom_root = DICOM_ROOT
        self.synthetic_root = SYNTHETIC_ROOT
        self.preview_root = PREVIEW_ROOT
        self.report_root = REPORT_ROOT

        for directory in [
            self.dicom_root,
            self.synthetic_root,
            self.preview_root,
            self.report_root,
        ]:
            directory.mkdir(parents=True, exist_ok=True)

        self.session = requests.Session()

        if ORTHANC_USERNAME and ORTHANC_PASSWORD:
            self.session.auth = (
                ORTHANC_USERNAME,
                ORTHANC_PASSWORD,
            )

    # ========================================================
    # 1. DISCOVER DICOM FILES
    # ========================================================

    def discover_dicoms(self, source=None):

        source = Path(source) if source else self.dicom_root

        if not source.exists():
            raise FileNotFoundError(source)

        files = [
            path for path in source.rglob("*")
            if path.is_file()
        ]

        print("\n========== DICOM DISCOVERY ==========")

        print("Source:", source)
        print("Files discovered:", len(files))

        return files

    # ========================================================
    # 2. VALIDATE DICOM FILES
    # ========================================================

    def validate_dicom(self, path, decode_pixels=True):

        result = {
            "file": str(path),
            "valid": False,
            "errors": [],
            "warnings": [],
            "metadata": {},
        }

        try:

            dataset = pydicom.dcmread(
                str(path),
                force=False,
            )

            required_fields = [
                "SOPClassUID",
                "SOPInstanceUID",
                "StudyInstanceUID",
                "SeriesInstanceUID",
                "Modality",
            ]

            for field in required_fields:

                if not hasattr(dataset, field):

                    result["errors"].append(
                        f"Missing required field: {field}"
                    )

            if hasattr(dataset, "Rows"):
                result["metadata"]["rows"] = int(dataset.Rows)

            if hasattr(dataset, "Columns"):
                result["metadata"]["columns"] = int(dataset.Columns)

            if hasattr(dataset, "Modality"):
                result["metadata"]["modality"] = str(
                    dataset.Modality
                )

            if hasattr(dataset, "PhotometricInterpretation"):
                result["metadata"]["photometric_interpretation"] = (
                    str(dataset.PhotometricInterpretation)
                )

            if hasattr(dataset, "StudyInstanceUID"):
                result["metadata"]["study_uid"] = str(
                    dataset.StudyInstanceUID
                )

            if hasattr(dataset, "SeriesInstanceUID"):
                result["metadata"]["series_uid"] = str(
                    dataset.SeriesInstanceUID
                )

            if not hasattr(dataset, "PixelData"):

                result["errors"].append(
                    "PixelData is missing"
                )

            elif decode_pixels:

                try:

                    pixels = dataset.pixel_array

                    if pixels.size == 0:

                        result["errors"].append(
                            "Decoded pixel array is empty"
                        )

                    else:

                        result["metadata"]["pixel_shape"] = (
                            list(pixels.shape)
                        )

                        result["metadata"]["pixel_dtype"] = (
                            str(pixels.dtype)
                        )

                except Exception as error:

                    result["errors"].append(
                        f"Pixel decoding failed: {error}"
                    )

                    result["warnings"].append(
                        "The required pixel codec may be unavailable."
                    )

            if not hasattr(dataset, "PatientID"):

                result["warnings"].append(
                    "PatientID is missing"
                )

            result["valid"] = len(result["errors"]) == 0

        except Exception as error:

            result["errors"].append(str(error))

        return result

    # ========================================================
    # 3. AUDIT DICOM DATASET
    # ========================================================

    def audit_dataset(self, source=None):

        print("\n========== DICOM DATASET AUDIT ==========")

        files = self.discover_dicoms(source)

        results = []

        for index, path in enumerate(files, start=1):

            print(
                f"\rValidating {index}/{len(files)}",
                end="",
                flush=True,
            )

            results.append(
                self.validate_dicom(path)
            )

        print("\n")

        valid_count = sum(
            item["valid"] for item in results
        )

        invalid_count = len(results) - valid_count

        report = {
            "created_at": datetime.now(
                timezone.utc
            ).isoformat(),

            "total_files": len(results),
            "valid_files": valid_count,
            "invalid_files": invalid_count,

            "results": results,
        }

        report_path = (
            self.report_root / "dicom_validation_report.json"
        )

        with open(report_path, "w") as file:

            json.dump(report, file, indent=4)

        print("Total files:", len(results))
        print("Valid files:", valid_count)
        print("Invalid files:", invalid_count)

        print("Report:", report_path)

        return report

    # ========================================================
    # 4. EXTRACT DICOM METADATA
    # ========================================================

    def extract_metadata(self, dicom_path):

        dataset = pydicom.dcmread(
            str(dicom_path),
            stop_before_pixels=True,
        )

        metadata = {
            "modality": str(
                getattr(dataset, "Modality", "UNKNOWN")
            ),

            "rows": int(
                getattr(dataset, "Rows", 0)
            ),

            "columns": int(
                getattr(dataset, "Columns", 0)
            ),

            "bits_allocated": int(
                getattr(dataset, "BitsAllocated", 0)
            ),

            "photometric_interpretation": str(
                getattr(
                    dataset,
                    "PhotometricInterpretation",
                    "UNKNOWN",
                )
            ),

            "study_instance_uid": str(
                getattr(dataset, "StudyInstanceUID", "")
            ),

            "series_instance_uid": str(
                getattr(dataset, "SeriesInstanceUID", "")
            ),
        }

        print("\n========== DICOM METADATA ==========")

        for key, value in metadata.items():

            print(f"{key}: {value}")

        return metadata

    # ========================================================
    # 5. EXTRACT AND PROCESS PIXEL DATA
    # ========================================================

    def extract_pixels(self, dicom_path):

        dataset = pydicom.dcmread(
            str(dicom_path)
        )

        if "PixelData" not in dataset:

            raise ValueError(
                "DICOM does not contain PixelData"
            )

        pixels = dataset.pixel_array

        return dataset, pixels

    # ========================================================
    # 6. CT WINDOWING
    # ========================================================

    @staticmethod
    def ct_windowing(
        pixels,
        window_center=None,
        window_width=None,
        rescale_slope=1.0,
        rescale_intercept=0.0,
    ):

        pixels = np.asarray(
            pixels,
            dtype=np.float32,
        )

        # Convert stored CT values to Hounsfield units.
        hu = (
            pixels * float(rescale_slope)
            + float(rescale_intercept)
        )

        if window_center is None or window_width is None:

            raise ValueError(
                "CT window center and width are required"
            )

        if float(window_width) <= 0:

            raise ValueError(
                "Window width must be positive"
            )

        center = float(window_center)
        width = float(window_width)

        lower = center - width / 2
        upper = center + width / 2

        windowed = np.clip(
            hu,
            lower,
            upper,
        )

        normalized = (
            (windowed - lower)
            / (upper - lower)
            * 255
        )

        return normalized.astype(np.uint8)

    # ========================================================
    # 7. CONVERT DICOM PIXELS TO PREVIEW
    # ========================================================

    def create_preview(
        self,
        dicom_path,
        output_path=None,
        window_center=None,
        window_width=None,
    ):

        dataset, pixels = self.extract_pixels(
            dicom_path
        )

        modality = str(
            getattr(dataset, "Modality", "")
        ).upper()

        if pixels.ndim not in (2, 3):

            raise ValueError(
                f"Unsupported pixel dimensions: {pixels.shape}"
            )

        if modality == "CT":

            if pixels.ndim != 2:

                raise ValueError(
                    "CT windowing currently expects 2D slices"
                )

            if window_center is None:
                window_center = getattr(
                    dataset, "WindowCenter", None
                )

            if window_width is None:
                window_width = getattr(
                    dataset, "WindowWidth", None
                )

            if isinstance(window_center, pydicom.multival.MultiValue):
                window_center = window_center[0]

            if isinstance(window_width, pydicom.multival.MultiValue):
                window_width = window_width[0]

            pixels = self.ct_windowing(
                pixels,
                window_center,
                window_width,
                getattr(dataset, "RescaleSlope", 1.0),
                getattr(dataset, "RescaleIntercept", 0.0),
            )

        elif pixels.ndim == 2:

            pixels = pixels.astype(np.float32)

            minimum = pixels.min()
            maximum = pixels.max()

            if maximum > minimum:

                pixels = (
                    (pixels - minimum)
                    / (maximum - minimum)
                    * 255
                )

            else:

                pixels = np.zeros_like(pixels)

            pixels = pixels.astype(np.uint8)

            if getattr(
                dataset,
                "PhotometricInterpretation",
                "",
            ) == "MONOCHROME1":

                pixels = 255 - pixels

        elif pixels.ndim == 3:

            if pixels.shape[-1] not in (3, 4):

                raise ValueError(
                    "Unsupported color pixel format"
                )

            pixels = pixels.astype(np.float32)

            if pixels.max() > 255:

                pixels = (
                    pixels / pixels.max() * 255
                )

            pixels = np.clip(
                pixels,
                0,
                255,
            ).astype(np.uint8)

        image = Image.fromarray(pixels)

        if output_path is None:

            output_path = (
                self.preview_root
                / f"{Path(dicom_path).stem}.png"
            )

        output_path = Path(output_path)

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        image.save(output_path)

        print("Preview saved:", output_path)

        return output_path

    # ========================================================
    # 8. CREATE SYNTHETIC DICOM FROM JPEG / PNG
    # ========================================================

    def create_synthetic_dicom(
        self,
        image_path,
        output_path=None,
    ):

        image_path = Path(image_path)

        if image_path.suffix.lower() not in (
            SUPPORTED_IMAGE_EXTENSIONS
        ):

            raise ValueError(
                "Only JPEG and PNG images are supported"
            )

        with Image.open(image_path) as image:

            image = image.convert("L")

            pixels = np.asarray(
                image,
                dtype=np.uint8,
            )

        rows, columns = pixels.shape

        sop_instance_uid = generate_uid()

        study_uid = generate_uid()

        series_uid = generate_uid()

        file_meta = FileMetaDataset()

        file_meta.MediaStorageSOPClassUID = (
            SecondaryCaptureImageStorage
        )

        file_meta.MediaStorageSOPInstanceUID = (
            sop_instance_uid
        )

        file_meta.TransferSyntaxUID = (
            ExplicitVRLittleEndian
        )

        file_meta.ImplementationClassUID = generate_uid()

        if output_path is None:

            output_path = (
                self.synthetic_root
                / f"{image_path.stem}_synthetic.dcm"
            )

        output_path = Path(output_path)

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        dataset = FileDataset(
            str(output_path),
            {},
            file_meta=file_meta,
            preamble=b"\0" * 128,
        )

        dataset.SpecificCharacterSet = "ISO_IR 192"

        dataset.SOPClassUID = SecondaryCaptureImageStorage

        dataset.SOPInstanceUID = sop_instance_uid

        dataset.StudyInstanceUID = study_uid

        dataset.SeriesInstanceUID = series_uid

        dataset.Modality = "OT"

        dataset.PatientName = "SYNTHETIC^TEST"

        dataset.PatientID = "SYNTHETIC-0001"

        dataset.StudyDate = datetime.now().strftime("%Y%m%d")

        dataset.StudyTime = datetime.now().strftime("%H%M%S")

        dataset.ImageType = [
            "DERIVED",
            "SECONDARY",
        ]

        dataset.ConversionType = "WSD"

        dataset.PatientIdentityRemoved = "YES"

        dataset.DeidentificationMethod = (
            "Synthetic test object generated from a JPEG image"
        )

        dataset.Rows = rows

        dataset.Columns = columns

        dataset.SamplesPerPixel = 1

        dataset.PhotometricInterpretation = "MONOCHROME2"

        dataset.BitsAllocated = 8

        dataset.BitsStored = 8

        dataset.HighBit = 7

        dataset.PixelRepresentation = 0

        dataset.PixelData = pixels.tobytes()

        dataset.is_little_endian = True

        dataset.is_implicit_VR = False

        dataset.save_as(
            str(output_path),
            enforce_file_format=True,
        )

        print("\nSynthetic DICOM created")

        print("Output:", output_path)

        print("Rows:", rows)

        print("Columns:", columns)

        print("Modality:", dataset.Modality)

        print(
            "Warning: This is a synthetic software-test "
            "object, not original clinical DICOM."
        )

        return output_path

    # ========================================================
    # 9. ORTHANC CONNECTION
    # ========================================================

    def orthanc_request(
        self,
        method,
        endpoint,
        **kwargs,
    ):

        url = (
            ORTHANC_URL
            + endpoint
        )

        response = self.session.request(
            method,
            url,
            timeout=REQUEST_TIMEOUT,
            **kwargs,
        )

        response.raise_for_status()

        return response

    def check_orthanc(self):

        print("\n========== ORTHANC HEALTH CHECK ==========")

        response = self.orthanc_request(
            "GET",
            "/system",
        )

        data = response.json()

        result = {
            "connected": True,
            "name": data.get("Name"),
            "version": data.get("Version"),
        }

        print("Connected:", result["connected"])

        print("Name:", result["name"])

        print("Version:", result["version"])

        return result

    # ========================================================
    # 10. UPLOAD DICOM TO ORTHANC
    # ========================================================

    def upload_to_orthanc(self, dicom_path):

        dicom_path = Path(dicom_path)

        validation = self.validate_dicom(
            dicom_path,
            decode_pixels=True,
        )

        if not validation["valid"]:

            raise ValueError(
                "Invalid DICOM cannot be uploaded: "
                + str(validation["errors"])
            )

        with open(dicom_path, "rb") as file:

            response = self.orthanc_request(
                "POST",
                "/instances",
                data=file,
                headers={
                    "Content-Type": "application/dicom"
                },
            )

        result = response.json()

        print("\n========== ORTHANC UPLOAD ==========")

        print("Status:", result.get("Status"))

        print("Instance ID:", result.get("ID"))

        print("Parent study:", result.get("ParentStudy"))

        return result

    # ========================================================
    # 11. LIST ORTHANC STUDIES
    # ========================================================

    def list_orthanc_studies(self):

        response = self.orthanc_request(
            "GET",
            "/studies",
        )

        studies = response.json()

        print("\n========== ORTHANC STUDIES ==========")

        print("Number of studies:", len(studies))

        return studies

    # ========================================================
    # 12. DOWNLOAD DICOM FROM ORTHANC
    # ========================================================

    def download_from_orthanc(
        self,
        instance_id,
        output_path=None,
    ):

        response = self.orthanc_request(
            "GET",
            f"/instances/{instance_id}/file",
        )

        if output_path is None:

            output_path = (
                self.dicom_root
                / f"{instance_id}.dcm"
            )

        output_path = Path(output_path)

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        output_path.write_bytes(
            response.content
        )

        print("DICOM downloaded:", output_path)

        return output_path


# ============================================================
# COMMAND LINE INTERFACE
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description="Phase 3 DICOM and Orthanc Pipeline"
    )

    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
    )

    # Audit
    audit_parser = subparsers.add_parser("audit")

    audit_parser.add_argument(
        "--source",
        default=str(DICOM_ROOT),
    )

    # Metadata
    metadata_parser = subparsers.add_parser("metadata")

    metadata_parser.add_argument("file")

    # Preview
    preview_parser = subparsers.add_parser("preview")

    preview_parser.add_argument("file")

    preview_parser.add_argument("--output")

    preview_parser.add_argument("--window-center", type=float)

    preview_parser.add_argument("--window-width", type=float)

    # Synthetic DICOM
    synthetic_parser = subparsers.add_parser("synthetic")

    synthetic_parser.add_argument("image")

    synthetic_parser.add_argument("--output")

    # Orthanc check
    subparsers.add_parser("orthanc-check")

    # Upload
    upload_parser = subparsers.add_parser("upload")

    upload_parser.add_argument("file")

    # List studies
    subparsers.add_parser("studies")

    # Download
    download_parser = subparsers.add_parser("download")

    download_parser.add_argument("instance_id")

    download_parser.add_argument("--output")

    args = parser.parse_args()

    pipeline = DICOMPipeline()

    try:

        if args.command == "audit":

            pipeline.audit_dataset(args.source)

        elif args.command == "metadata":

            pipeline.extract_metadata(args.file)

        elif args.command == "preview":

            pipeline.create_preview(
                args.file,
                args.output,
                args.window_center,
                args.window_width,
            )

        elif args.command == "synthetic":

            pipeline.create_synthetic_dicom(
                args.image,
                args.output,
            )

        elif args.command == "orthanc-check":

            pipeline.check_orthanc()

        elif args.command == "upload":

            pipeline.upload_to_orthanc(
                args.file
            )

        elif args.command == "studies":

            pipeline.list_orthanc_studies()

        elif args.command == "download":

            pipeline.download_from_orthanc(
                args.instance_id,
                args.output,
            )

    except requests.exceptions.ConnectionError:

        print(
            "ERROR: Cannot connect to Orthanc. "
            "Check that the server is running."
        )

        sys.exit(1)

    except requests.exceptions.HTTPError as error:

        print("Orthanc HTTP error:", error)

        sys.exit(1)

    except Exception as error:

        print("ERROR:", error)

        sys.exit(1)


if __name__ == "__main__":

    main()
