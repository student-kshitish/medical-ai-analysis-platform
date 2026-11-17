"""
PHASE 10: FASTAPI BACKEND

Connects the React frontend to the Phase 9 inference engine.

Endpoints:
    GET  /                 API information
    GET  /health           Backend and model health
    POST /api/v1/predict   X-ray prediction

Research prototype only. Not for clinical diagnosis.
"""

import json
import logging
import sys
import uuid

from contextlib import asynccontextmanager
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path

from PIL import Image, UnidentifiedImageError
from phase13_doctor_review import router as doctor_review_router

from phase11_database import (
    SessionLocal,
    User,
    Patient,
    Scan,
    Prediction,
    AuditLog,
    hash_password,
)
from fastapi import (
    FastAPI,
    File,
    Form,
    HTTPException,
    UploadFile,
)

from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from starlette.concurrency import run_in_threadpool


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

AIML_DIR = PROJECT_ROOT / "AIML-01"

UPLOAD_DIR = PROJECT_ROOT / "AIML-02/storage/uploads"

PREDICTION_DIR = (
    PROJECT_ROOT / "AIML-02/storage/predictions"
)

ARTIFACT_DIR = PROJECT_ROOT / "artifacts"

MAX_UPLOAD_BYTES = 15 * 1024 * 1024

MAX_IMAGE_PIXELS = 25_000_000

ALLOWED_FORMATS = {
    "JPEG": ".jpg",
    "PNG": ".png",
}


# ============================================================
# DIRECTORY SETUP
# ============================================================

UPLOAD_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

PREDICTION_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

ARTIFACT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger("medical-ai-backend")
def save_prediction_to_database(prediction, upload):
    """
    Save synthetic research patient, scan, prediction and audit record.
    """

    db = SessionLocal()

    try:
        # 1. System uploader
        uploader = db.query(User).filter(
            User.email == "research-uploader@local.invalid"
        ).first()

        if uploader is None:
            uploader = User(
                email="research-uploader@local.invalid",
                full_name="Research System Uploader",
                password_hash=hash_password(str(uuid.uuid4())),
                role="system",
                is_active=True,
            )

            db.add(uploader)
            db.flush()

        # 2. Synthetic research patient
        patient = Patient(
            patient_code=f"DEMO-{prediction['request_id']}",
            display_name="SYNTHETIC RESEARCH DEMO - NOT A REAL PATIENT",
        )

        db.add(patient)
        db.flush()

        # 3. Scan record
        scan = Scan(
            patient_id=patient.id,
            uploaded_by=uploader.id,
            original_filename=upload["original_filename"],
            storage_path=str(upload["path"]),
            modality="XRAY",
            file_format=upload["metadata"]["format"],
            status="pending_review",
        )

        db.add(scan)
        db.flush()

        # 4. AI prediction record
        probabilities = prediction["probabilities"]

        prediction_record = Prediction(
            scan_id=scan.id,
            predicted_class=prediction["prediction"],
            confidence=float(max(probabilities.values())),
            probabilities=probabilities,
            model_version=(
                f"phase6-epoch-{prediction.get('model_epoch', 'unknown')}"
            ),
            explanation_path=prediction.get(
                "explanation_urls", {}
            ).get("overlay"),
        )

        db.add(prediction_record)
        db.flush()

        # 5. Audit record
        audit = AuditLog(
            actor_id=uploader.id,
            action="ai_prediction_created",
            resource_type="scan",
            resource_id=str(scan.id),
            details={
                "request_id": prediction["request_id"],
                "scan_id": scan.id,
                "prediction_id": prediction_record.id,
                "model_epoch": prediction.get("model_epoch"),
                "synthetic_demo": True,
            },
        )

        db.add(audit)

        # 6. Commit transaction
        db.commit()

        return {
            "patient_id": patient.id,
            "scan_id": scan.id,
            "prediction_id": prediction_record.id,
            "database_status": "saved",
        }

    except Exception:
        db.rollback()
        logger.exception("Database prediction storage failed")
        raise

    finally:
        db.close()

# ============================================================
# IMPORT PHASE 9 INFERENCE ENGINE
# ============================================================

sys.path.insert(
    0,
    str(AIML_DIR),
)

from phase9_inference import MedicalInferenceEngine


# ============================================================
# GLOBAL ENGINE
# ============================================================

inference_engine = None


# ============================================================
# APPLICATION LIFESPAN
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):

    global inference_engine

    logger.info("Starting Medical AI Backend")

    try:
        inference_engine = MedicalInferenceEngine()

        logger.info(
            "Phase 9 inference engine initialized."
        )

    except Exception:
        logger.exception(
            "Failed to initialize inference engine."
        )
        raise

    yield

    logger.info("Shutting down Medical AI Backend")

    inference_engine = None


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="Medical AI Analysis API",
    description=(
        "Research prototype for chest X-ray image "
        "classification and explainability."
    ),
    version="1.0.0",
    lifespan=lifespan,
)
app.include_router(doctor_review_router)

# ============================================================
# CORS CONFIGURATION
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=[
        "GET",
        "POST",
    ],
    allow_headers=["*"],
)


# ============================================================
# STATIC FILE ACCESS
# ============================================================

app.mount(
    "/uploads",
    StaticFiles(directory=str(UPLOAD_DIR)),
    name="uploads",
)

app.mount(
    "/artifacts",
    StaticFiles(directory=str(ARTIFACT_DIR)),
    name="artifacts",
)


# ============================================================
# ROOT ENDPOINT
# ============================================================

@app.get("/")
async def root():

    return {
        "application": "Medical AI Analysis Platform",
        "version": "1.0.0",
        "status": "running",
        "api_documentation": "/docs",
        "health_endpoint": "/health",
        "prediction_endpoint": "/api/v1/predict",
        "warning": (
            "Research prototype only. "
            "Not a medical diagnosis."
        ),
    }


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
async def health():

    if inference_engine is None:

        return {
            "status": "unavailable",
            "model_loaded": False,
        }

    return {
        "status": "healthy",
        "model_loaded": True,
        "model_epoch": inference_engine.epoch,
        "classes": inference_engine.class_names,
        "device": str(inference_engine.device),
        "gpu_available": (
            inference_engine.device.type == "cuda"
        ),
        "timestamp": datetime.now(
            timezone.utc
        ).isoformat(),
    }


# ============================================================
# IMAGE VALIDATION
# ============================================================

def validate_image(image_bytes):
    """
    Validate actual image contents using Pillow.

    We do not trust only the uploaded filename or MIME type.
    """

    try:

        with Image.open(
            BytesIO(image_bytes)
        ) as image:

            image_format = image.format

            width, height = image.size

            if image_format not in ALLOWED_FORMATS:
                raise HTTPException(
                    status_code=415,
                    detail=(
                        "Unsupported image format. "
                        "Upload JPEG or PNG."
                    ),
                )

            if width <= 0 or height <= 0:
                raise HTTPException(
                    status_code=400,
                    detail="Invalid image dimensions.",
                )

            if width * height > MAX_IMAGE_PIXELS:
                raise HTTPException(
                    status_code=413,
                    detail="Image dimensions are too large.",
                )

            image.verify()

        return {
            "format": image_format,
            "width": width,
            "height": height,
            "extension": ALLOWED_FORMATS[image_format],
        }

    except HTTPException:
        raise

    except (
        UnidentifiedImageError,
        OSError,
        Image.DecompressionBombError,
    ) as error:

        raise HTTPException(
            status_code=400,
            detail=f"Invalid or corrupted image: {error}",
        )


# ============================================================
# SAVE UPLOADED IMAGE
# ============================================================

async def save_upload(file: UploadFile):

    if inference_engine is None:
        raise HTTPException(
            status_code=503,
            detail="Inference engine is not available.",
        )

    image_bytes = await file.read(
        MAX_UPLOAD_BYTES + 1
    )

    if not image_bytes:
        raise HTTPException(
            status_code=400,
            detail="Uploaded file is empty.",
        )

    if len(image_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail="Maximum upload size is 15 MB.",
        )

    metadata = validate_image(image_bytes)

    request_id = str(uuid.uuid4())

    filename = (
        request_id
        + metadata["extension"]
    )

    destination = UPLOAD_DIR / filename

    with open(destination, "wb") as output_file:
        output_file.write(image_bytes)

    return {
        "request_id": request_id,
        "filename": filename,
        "path": destination,
        "original_filename": Path(
            file.filename or "uploaded_image"
        ).name,
        "metadata": metadata,
    }


# ============================================================
# SAVE PREDICTION JSON
# ============================================================

def save_prediction(result):

    request_id = result["request_id"]

    output_path = (
        PREDICTION_DIR
        / f"{request_id}.json"
    )

    temporary_path = output_path.with_suffix(".tmp")

    with open(temporary_path, "w") as file:

        json.dump(
            result,
            file,
            indent=4,
        )

    temporary_path.replace(output_path)

    return output_path


# ============================================================
# PREDICTION ENDPOINT
# ============================================================

@app.post("/api/v1/predict")
async def predict(
    file: UploadFile = File(...),
    explain: bool = Form(False),
):
    """
    Accept one JPEG or PNG image.

    Returns:
        Predicted class
        Class probabilities
        Model information
        Optional Grad-CAM URLs
    """

    if inference_engine is None:

        raise HTTPException(
            status_code=503,
            detail="Model is not ready.",
        )

    upload = await save_upload(file)

    request_id = upload["request_id"]

    logger.info(
        "Received image request: %s",
        request_id,
    )

    try:

        # Run the synchronous PyTorch inference outside
        # the asynchronous event loop.
        prediction = await run_in_threadpool(
            inference_engine.run,
            str(upload["path"]),
            explain,
        )

        prediction["request_id"] = request_id

        prediction["uploaded_filename"] = (
            upload["original_filename"]
        )

        prediction["image_metadata"] = {
            "format": upload["metadata"]["format"],
            "width": upload["metadata"]["width"],
            "height": upload["metadata"]["height"],
        }

        # Replace the server's internal filesystem path
        # with a URL that the frontend can access.
        prediction["image"] = (
            f"/uploads/{upload['filename']}"
        )

        # Convert Grad-CAM file paths to static URLs.
        if prediction.get("explanation_generated"):

            files = prediction.get(
                "explanation_files",
                {},
            )

            prediction["explanation_urls"] = {
                name: (
                    "/artifacts/"
                    + str(Path(path).resolve().relative_to(
                        ARTIFACT_DIR.resolve()
                    )).replace("\\", "/")
                )
                for name, path in files.items()
            }

            # Do not expose internal filesystem paths.
            prediction.pop(
                "explanation_files",
                None,
            )

        prediction["api_timestamp"] = datetime.now(
            timezone.utc
        ).isoformat()

        prediction["clinical_warning"] = (
            "Experimental research output only. "
            "This is not a medical diagnosis. "
            "A qualified doctor must review the image."
        )

        # Save scan, AI prediction and audit record in PostgreSQL
        database_records = save_prediction_to_database(
            prediction,
            upload,
        )

        # Include database IDs in API response
        prediction["database_records"] = database_records

        # Save prediction JSON report
        output_path = save_prediction(
             prediction
        )

        prediction["saved_report"] = output_path.name

        logger.info(
            "Prediction completed: %s",
            request_id,
        )

        return {
            "success": True,
            "data": prediction,
        }

    except HTTPException:
        raise

    except Exception:

        logger.exception(
            "Prediction failed: %s",
            request_id,
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "Image inference failed. "
                "Check the backend logs."
            ),
        )


# ============================================================
# SERVER ENTRY POINT
# ============================================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8000,
    )
