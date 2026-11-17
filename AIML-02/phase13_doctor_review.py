"""
PHASE 13 — DOCTOR REVIEW SYSTEM

Research prototype only.
The doctor's review is stored separately from the AI prediction.
"""

from datetime import datetime, timezone
from typing import Literal, Optional
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from phase11_database import (
    SessionLocal,
    User,
    Scan,
    Prediction,
    DoctorReview,
    AuditLog,
)


router = APIRouter(
    prefix="/api/v1/doctor",
    tags=["Doctor Review"],
)


# ============================================================
# REQUEST SCHEMA
# ============================================================

class ReviewRequest(BaseModel):

    decision: Literal[
        "approved",
        "rejected",
        "needs_review",
    ]

    comments: str = Field(
        default="",
        max_length=5000,
    )


# ============================================================
# DATABASE HELPERS
# ============================================================

def get_database():

    db = SessionLocal()

    try:
        yield db

    finally:
        db.close()


def serialize_datetime(value):

    if value is None:
        return None

    return value.isoformat()


def verify_doctor(db: Session, doctor_id: int):

    doctor = db.query(User).filter(
        User.id == doctor_id
    ).first()

    if doctor is None:
        raise HTTPException(
            status_code=404,
            detail="Doctor account not found.",
        )

    if not doctor.is_active:
        raise HTTPException(
            status_code=403,
            detail="Doctor account is inactive.",
        )

    if doctor.role not in {
        "doctor",
        "radiologist",
        "admin",
    }:
        raise HTTPException(
            status_code=403,
            detail="This account cannot perform doctor reviews.",
        )

    return doctor


def prediction_details(prediction):

    return {
        "prediction_id": prediction.id,
        "predicted_class": prediction.predicted_class,
        "confidence": prediction.confidence,
        "probabilities": prediction.probabilities,
        "model_version": prediction.model_version,
        "explanation_available": bool(
            prediction.explanation_path
        ),
    }


def scan_details(scan):

    return {
        "scan_id": scan.id,
        "patient_id": scan.patient_id,
        "original_filename": scan.original_filename,
        "modality": scan.modality,
        "file_format": scan.file_format,
        "status": scan.status,
        "created_at": serialize_datetime(scan.created_at),
    }


# ============================================================
# 1. DOCTOR REVIEW QUEUE
# ============================================================

@router.get("/reviews/pending")
def get_pending_reviews(
    doctor_id: int,
    db: Session = Depends(get_database),
):

    verify_doctor(db, doctor_id)

    rows = (
        db.query(Scan, Prediction)
        .join(
            Prediction,
            Prediction.scan_id == Scan.id,
        )
        .order_by(Scan.created_at.desc())
        .limit(200)
        .all()
    )

    pending = []

    for scan, prediction in rows:

        latest_review = (
            db.query(DoctorReview)
            .filter(
                DoctorReview.scan_id == scan.id
            )
            .order_by(
                DoctorReview.created_at.desc()
            )
            .first()
        )

        if latest_review is not None:
            if latest_review.review_status != "needs_review":
                continue

        pending.append({
            "scan": scan_details(scan),
            "prediction": prediction_details(prediction),
            "review_status": (
                latest_review.review_status
                if latest_review else "pending"
            ),
        })

    return {
        "total": len(pending),
        "reviews": pending,
    }



# ============================================================
# 2. GET SCAN AND AI PREDICTION DETAILS
# ============================================================

@router.get("/scans/{scan_id}")
def get_scan_for_review(
    scan_id: int,
    doctor_id: int,
):

    db = SessionLocal()

    try:

        verify_doctor(db, doctor_id)

        scan = db.query(Scan).filter(
            Scan.id == scan_id
        ).first()

        if scan is None:
            raise HTTPException(
                status_code=404,
                detail="Scan not found.",
            )

        prediction = (
            db.query(Prediction)
            .filter(
                Prediction.scan_id == scan_id
            )
            .order_by(Prediction.created_at.desc())
            .first()
        )

        if prediction is None:
            raise HTTPException(
                status_code=404,
                detail="No AI prediction exists for this scan.",
            )

        return {
            "scan": scan_details(scan),
            "prediction": prediction_details(prediction),
            "notice": (
                "AI output is an automated research result, "
                "not a confirmed medical diagnosis."
            ),
        }

    finally:
        db.close()


# ============================================================
# 3. SUBMIT DOCTOR REVIEW
# ============================================================

@router.post("/scans/{scan_id}/review")
def submit_doctor_review(
    scan_id: int,
    request: ReviewRequest,
    doctor_id: int,
):

    db = SessionLocal()

    try:

        doctor = verify_doctor(db, doctor_id)

        scan = db.query(Scan).filter(
            Scan.id == scan_id
        ).first()

        if scan is None:
            raise HTTPException(
                status_code=404,
                detail="Scan not found.",
            )

        prediction = (
            db.query(Prediction)
            .filter(
                Prediction.scan_id == scan_id
            )
            .order_by(Prediction.created_at.desc())
            .first()
        )

        if prediction is None:
            raise HTTPException(
                status_code=400,
                detail="Cannot review a scan without an AI prediction.",
            )

        # Create a new review record.
        # Previous reviews are retained as history.

        review = DoctorReview(
            scan_id=scan.id,
            prediction_id=prediction.id,
            reviewer_id=doctor.id,
            review_status=request.decision,
            comments=request.comments.strip(),
        )

        db.add(review)
        db.flush()

        # Record the review action in the audit log.

        audit = AuditLog(
            actor_id=doctor.id,
            action="doctor_review_submitted",
            resource_type="scan",
            resource_id=str(scan.id),
            details={
                "review_id": review.id,
                "decision": request.decision,
                "prediction_id": prediction.id,
            },
        )

        db.add(audit)
        db.commit()
        db.refresh(review)

        return {
            "success": True,
            "message": "Doctor review saved successfully.",
            "review": {
                "review_id": review.id,
                "scan_id": review.scan_id,
                "prediction_id": review.prediction_id,
                "reviewer_id": review.reviewer_id,
                "decision": review.review_status,
                "comments": review.comments,
                "created_at": serialize_datetime(
                    review.created_at
                ),
            },
        }

    except Exception:
        db.rollback()
        raise

    finally:
        db.close()


# ============================================================
# 4. REVIEW HISTORY
# ============================================================

@router.get("/scans/{scan_id}/history")
def get_review_history(
    scan_id: int,
    doctor_id: int,
):

    db = SessionLocal()

    try:

        verify_doctor(db, doctor_id)

        scan = db.query(Scan).filter(
            Scan.id == scan_id
        ).first()

        if scan is None:
            raise HTTPException(
                status_code=404,
                detail="Scan not found.",
            )

        reviews = (
            db.query(DoctorReview, User)
            .join(
                User,
                User.id == DoctorReview.reviewer_id,
            )
            .filter(
                DoctorReview.scan_id == scan_id
            )
            .order_by(
                DoctorReview.created_at.desc()
            )
            .all()
        )

        history = []

        for review, reviewer in reviews:

            history.append({
                "review_id": review.id,
                "decision": review.review_status,
                "comments": review.comments,
                "reviewer": reviewer.full_name,
                "reviewer_id": reviewer.id,
                "created_at": serialize_datetime(
                    review.created_at
                ),
            })

        return {
            "scan_id": scan_id,
            "total_reviews": len(history),
            "history": history,
        }

    finally:
        db.close()


# ============================================================
# HEALTH CHECK
# ============================================================

@router.get("/health")
def doctor_review_health():

    return {
        "service": "doctor-review",
        "status": "healthy",
        "timestamp": datetime.now(
            timezone.utc
        ).isoformat(),
    }


if __name__ == "__main__":

    print("Phase 13 Doctor Review System")
    print("Router created successfully.")
    print("Register this router with the Phase 10 FastAPI app.")
