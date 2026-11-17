"""
PHASE 11: POSTGRESQL DATABASE INTEGRATION

Database:
    medical_ai_db

Tables:
    users
    patients
    scans
    predictions
    doctor_reviews
    audit_logs

Features:
    - SQLAlchemy ORM
    - PostgreSQL connection
    - Secure password hashing
    - User creation and authentication
    - Patient management
    - Scan metadata management
    - AI prediction storage
    - Doctor review storage
    - Audit logging
    - Database health checks

Research prototype. Do not store real patient data yet.
"""

import os
import sys
import json
import hmac
import hashlib
import secrets
import argparse

from pathlib import Path
from datetime import datetime, date, timezone

from dotenv import load_dotenv

from sqlalchemy import (
    create_engine,
    String,
    Integer,
    Float,
    Date,
    DateTime,
    ForeignKey,
    Text,
    JSON,
    Boolean,
    select,
    func,
    URL,
)

from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    mapped_column,
    relationship,
    sessionmaker,
    Session,
)

from sqlalchemy.exc import SQLAlchemyError


# ============================================================
# PROJECT CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

load_dotenv(PROJECT_ROOT / ".env")

DATABASE_HOST = os.getenv("DB_HOST", "127.0.0.1")
DATABASE_PORT = int(os.getenv("DB_PORT", "5432"))
DATABASE_NAME = os.getenv("DB_NAME", "medical_ai_db")
DATABASE_USER = os.getenv("DB_USER", "medical_ai_user")
DATABASE_PASSWORD = os.getenv("DB_PASSWORD")

if not DATABASE_PASSWORD:
    raise RuntimeError(
        "DB_PASSWORD is missing. Configure it in the project .env file."
    )


# ============================================================
# DATABASE CONNECTION
# ============================================================

DATABASE_URL = URL.create(
    drivername="postgresql+psycopg2",
    username=DATABASE_USER,
    password=DATABASE_PASSWORD,
    host=DATABASE_HOST,
    port=DATABASE_PORT,
    database=DATABASE_NAME,
)

engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
    pool_size=5,
    max_overflow=10,
    echo=False,
)

SessionLocal = sessionmaker(
    bind=engine,
    autoflush=False,
    expire_on_commit=False,
)


# ============================================================
# BASE MODEL
# ============================================================

class Base(DeclarativeBase):
    pass


# ============================================================
# TABLE 1: USERS
# ============================================================

class User(Base):

    __tablename__ = "users"

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        index=True,
    )

    email: Mapped[str] = mapped_column(
        String(255),
        unique=True,
        index=True,
        nullable=False,
    )

    full_name: Mapped[str] = mapped_column(
        String(150),
        nullable=False,
    )

    password_hash: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
    )

    role: Mapped[str] = mapped_column(
        String(30),
        default="doctor",
        nullable=False,
    )

    is_active: Mapped[bool] = mapped_column(
        Boolean,
        default=True,
        nullable=False,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    reviews: Mapped[list["DoctorReview"]] = relationship(
        back_populates="reviewer",
        foreign_keys="DoctorReview.reviewer_id",
    )

    uploaded_scans: Mapped[list["Scan"]] = relationship(
        back_populates="uploader",
        foreign_keys="Scan.uploaded_by",
    )

    audit_entries: Mapped[list["AuditLog"]] = relationship(
        back_populates="actor",
    )


# ============================================================
# TABLE 2: PATIENTS
# ============================================================

class Patient(Base):

    __tablename__ = "patients"

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        index=True,
    )

    patient_code: Mapped[str] = mapped_column(
        String(80),
        unique=True,
        index=True,
        nullable=False,
    )

    display_name: Mapped[str] = mapped_column(
        String(150),
        nullable=False,
    )

    date_of_birth: Mapped[date | None] = mapped_column(
        Date,
        nullable=True,
    )

    sex: Mapped[str | None] = mapped_column(
        String(20),
        nullable=True,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    scans: Mapped[list["Scan"]] = relationship(
        back_populates="patient",
    )


# ============================================================
# TABLE 3: SCANS
# ============================================================

class Scan(Base):

    __tablename__ = "scans"

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        index=True,
    )

    patient_id: Mapped[int] = mapped_column(
        ForeignKey("patients.id", ondelete="RESTRICT"),
        nullable=False,
        index=True,
    )

    uploaded_by: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="RESTRICT"),
        nullable=False,
        index=True,
    )

    original_filename: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
    )

    storage_path: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    modality: Mapped[str] = mapped_column(
        String(30),
        default="XRAY",
        nullable=False,
    )

    file_format: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
    )

    status: Mapped[str] = mapped_column(
        String(30),
        default="uploaded",
        nullable=False,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    patient: Mapped["Patient"] = relationship(
        back_populates="scans",
    )

    uploader: Mapped["User"] = relationship(
        back_populates="uploaded_scans",
        foreign_keys=[uploaded_by],
    )

    predictions: Mapped[list["Prediction"]] = relationship(
        back_populates="scan",
    )

    reviews: Mapped[list["DoctorReview"]] = relationship(
        back_populates="scan",
    )


# ============================================================
# TABLE 4: PREDICTIONS
# ============================================================

class Prediction(Base):

    __tablename__ = "predictions"

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        index=True,
    )

    scan_id: Mapped[int] = mapped_column(
        ForeignKey("scans.id", ondelete="RESTRICT"),
        nullable=False,
        index=True,
    )

    predicted_class: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
    )

    confidence: Mapped[float] = mapped_column(
        Float,
        nullable=False,
    )

    probabilities: Mapped[dict] = mapped_column(
        JSON,
        nullable=False,
    )

    model_version: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
    )

    explanation_path: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    scan: Mapped["Scan"] = relationship(
        back_populates="predictions",
    )

    reviews: Mapped[list["DoctorReview"]] = relationship(
        back_populates="prediction",
    )


# ============================================================
# TABLE 5: DOCTOR REVIEWS
# ============================================================

class DoctorReview(Base):

    __tablename__ = "doctor_reviews"

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        index=True,
    )

    scan_id: Mapped[int] = mapped_column(
        ForeignKey("scans.id", ondelete="RESTRICT"),
        nullable=False,
        index=True,
    )

    prediction_id: Mapped[int | None] = mapped_column(
        ForeignKey("predictions.id", ondelete="RESTRICT"),
        nullable=True,
    )

    reviewer_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="RESTRICT"),
        nullable=False,
        index=True,
    )

    review_status: Mapped[str] = mapped_column(
        String(30),
        default="pending",
        nullable=False,
    )

    comments: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    scan: Mapped["Scan"] = relationship(
        back_populates="reviews",
    )

    prediction: Mapped["Prediction"] = relationship(
        back_populates="reviews",
    )

    reviewer: Mapped["User"] = relationship(
        back_populates="reviews",
        foreign_keys=[reviewer_id],
    )


# ============================================================
# TABLE 6: AUDIT LOGS
# ============================================================

class AuditLog(Base):

    __tablename__ = "audit_logs"

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        index=True,
    )

    actor_id: Mapped[int | None] = mapped_column(
        ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )

    action: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
    )

    resource_type: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
    )

    resource_id: Mapped[str | None] = mapped_column(
        String(100),
        nullable=True,
    )

    details: Mapped[dict] = mapped_column(
        JSON,
        default=dict,
        nullable=False,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    actor: Mapped["User | None"] = relationship(
        back_populates="audit_entries",
    )


# ============================================================
# PASSWORD SECURITY
# ============================================================

def hash_password(password: str) -> str:
    """
    Hash a password using scrypt and a random salt.

    Never store the original password.
    """

    if len(password) < 12:
        raise ValueError(
            "Password must contain at least 12 characters."
        )

    salt = secrets.token_bytes(16)

    password_hash = hashlib.scrypt(
        password.encode("utf-8"),
        salt=salt,
        n=2**14,
        r=8,
        p=1,
        dklen=32,
    )

    return (
        salt.hex()
        + "$"
        + password_hash.hex()
    )


def verify_password(
    password: str,
    stored_hash: str,
) -> bool:
    """
    Verify a password against its stored scrypt hash.
    """

    try:

        salt_hex, hash_hex = stored_hash.split("$")

        salt = bytes.fromhex(salt_hex)

        expected_hash = bytes.fromhex(hash_hex)

        actual_hash = hashlib.scrypt(
            password.encode("utf-8"),
            salt=salt,
            n=2**14,
            r=8,
            p=1,
            dklen=32,
        )

        return hmac.compare_digest(
            actual_hash,
            expected_hash,
        )

    except (ValueError, TypeError):
        return False


# ============================================================
# DATABASE MANAGER
# ============================================================

class DatabaseManager:

    def __init__(self):
        self.engine = engine
        self.session_factory = SessionLocal

    # --------------------------------------------------------
    # INITIALIZE DATABASE
    # --------------------------------------------------------

    def initialize(self):

        print("\n========== DATABASE INITIALIZATION ==========")

        Base.metadata.create_all(
            bind=self.engine
        )

        print("Database tables created successfully.")

        print("\nTables:")

        for table_name in Base.metadata.tables:
            print(" -", table_name)

    # --------------------------------------------------------
    # CONNECTION TEST
    # --------------------------------------------------------

    def check_connection(self):

        with self.engine.connect() as connection:

            result = connection.execute(
                select(func.current_database())
            )

            database_name = result.scalar_one()

            result = connection.execute(
                select(func.current_user())
            )

            database_user = result.scalar_one()

        print("\n========== DATABASE CONNECTION ==========")
        print("Connection: SUCCESS")
        print("Database:", database_name)
        print("User:", database_user)

        return True

    # --------------------------------------------------------
    # CREATE USER
    # --------------------------------------------------------

    def create_user(
        self,
        email,
        full_name,
        password,
        role="doctor",
    ):

        allowed_roles = {
            "doctor",
            "admin",
            "radiologist",
        }

        email = email.strip().lower()

        if role not in allowed_roles:
            raise ValueError("Invalid user role.")

        if not email or "@" not in email:
            raise ValueError("Invalid email address.")

        password_hash = hash_password(password)

        with self.session_factory.begin() as session:

            existing = session.scalar(
                select(User).where(
                    User.email == email
                )
            )

            if existing:
                raise ValueError(
                    "An account with this email already exists."
                )

            user = User(
                email=email,
                full_name=full_name.strip(),
                password_hash=password_hash,
                role=role,
            )

            session.add(user)
            session.flush()

            user_id = user.id

            session.add(
                AuditLog(
                    actor_id=user_id,
                    action="USER_CREATED",
                    resource_type="USER",
                    resource_id=str(user_id),
                    details={
                        "role": role,
                    },
                )
            )

        print("User created. ID:", user_id)

        return user_id

    # --------------------------------------------------------
    # AUTHENTICATE USER
    # --------------------------------------------------------

    def authenticate_user(
        self,
        email,
        password,
    ):

        email = email.strip().lower()

        with self.session_factory() as session:

            user = session.scalar(
                select(User).where(
                    User.email == email
                )
            )

            if not user:
                return None

            if not user.is_active:
                return None

            if not verify_password(
                password,
                user.password_hash,
            ):
                return None

            return {
                "id": user.id,
                "email": user.email,
                "full_name": user.full_name,
                "role": user.role,
            }

    # --------------------------------------------------------
    # CREATE PATIENT
    # --------------------------------------------------------

    def create_patient(
        self,
        patient_code,
        display_name,
        date_of_birth=None,
        sex=None,
    ):

        with self.session_factory.begin() as session:

            existing = session.scalar(
                select(Patient).where(
                    Patient.patient_code == patient_code
                )
            )

            if existing:
                raise ValueError(
                    "Patient code already exists."
                )

            patient = Patient(
                patient_code=patient_code,
                display_name=display_name,
                date_of_birth=date_of_birth,
                sex=sex,
            )

            session.add(patient)
            session.flush()

            patient_id = patient.id

        print("Patient created. ID:", patient_id)

        return patient_id

    # --------------------------------------------------------
    # CREATE SCAN
    # --------------------------------------------------------

    def create_scan(
        self,
        patient_id,
        uploaded_by,
        original_filename,
        storage_path,
        file_format="JPEG",
        modality="XRAY",
    ):

        with self.session_factory.begin() as session:

            patient = session.get(
                Patient,
                patient_id,
            )

            uploader = session.get(
                User,
                uploaded_by,
            )

            if not patient:
                raise ValueError("Patient not found.")

            if not uploader or not uploader.is_active:
                raise ValueError("Active uploader not found.")

            scan = Scan(
                patient_id=patient_id,
                uploaded_by=uploaded_by,
                original_filename=original_filename,
                storage_path=storage_path,
                file_format=file_format,
                modality=modality,
                status="uploaded",
            )

            session.add(scan)
            session.flush()

            scan_id = scan.id

            session.add(
                AuditLog(
                    actor_id=uploaded_by,
                    action="SCAN_UPLOADED",
                    resource_type="SCAN",
                    resource_id=str(scan_id),
                    details={
                        "modality": modality,
                        "file_format": file_format,
                    },
                )
            )

        print("Scan created. ID:", scan_id)

        return scan_id

    # --------------------------------------------------------
    # SAVE AI PREDICTION
    # --------------------------------------------------------

    def create_prediction(
        self,
        scan_id,
        predicted_class,
        probabilities,
        model_version,
        explanation_path=None,
    ):

        if not isinstance(probabilities, dict):
            raise ValueError(
                "Probabilities must be a dictionary."
            )

        if not probabilities:
            raise ValueError(
                "Probabilities cannot be empty."
            )

        values = list(probabilities.values())

        if any(
            not isinstance(value, (int, float))
            or not 0 <= value <= 1
            for value in values
        ):
            raise ValueError(
                "Probabilities must be between 0 and 1."
            )

        if abs(sum(values) - 1.0) > 0.001:
            raise ValueError(
                "Probabilities must sum to approximately 1."
            )

        if predicted_class not in probabilities:
            raise ValueError(
                "Predicted class must exist in probabilities."
            )

        confidence = float(
            probabilities[predicted_class]
        )

        with self.session_factory.begin() as session:

            scan = session.get(
                Scan,
                scan_id,
            )

            if not scan:
                raise ValueError("Scan not found.")

            prediction = Prediction(
                scan_id=scan_id,
                predicted_class=predicted_class,
                confidence=confidence,
                probabilities=probabilities,
                model_version=model_version,
                explanation_path=explanation_path,
            )

            session.add(prediction)

            scan.status = "analyzed"

            session.flush()

            prediction_id = prediction.id

            session.add(
                AuditLog(
                    actor_id=None,
                    action="AI_PREDICTION_CREATED",
                    resource_type="PREDICTION",
                    resource_id=str(prediction_id),
                    details={
                        "predicted_class": predicted_class,
                        "model_version": model_version,
                    },
                )
            )

        print(
            "Prediction created. ID:",
            prediction_id,
        )

        return prediction_id

    # --------------------------------------------------------
    # CREATE DOCTOR REVIEW
    # --------------------------------------------------------

    def create_review(
        self,
        scan_id,
        reviewer_id,
        comments,
        prediction_id=None,
        review_status="reviewed",
    ):

        allowed_statuses = {
            "pending",
            "reviewed",
            "flagged",
        }

        if review_status not in allowed_statuses:
            raise ValueError("Invalid review status.")

        with self.session_factory.begin() as session:

            scan = session.get(
                Scan,
                scan_id,
            )

            reviewer = session.get(
                User,
                reviewer_id,
            )

            if not scan:
                raise ValueError("Scan not found.")

            if not reviewer or not reviewer.is_active:
                raise ValueError("Active reviewer not found.")

            if reviewer.role not in {
                "doctor",
                "radiologist",
                "admin",
            }:
                raise ValueError(
                    "User is not permitted to review scans."
                )

            if prediction_id is not None:

                prediction = session.get(
                    Prediction,
                    prediction_id,
                )

                if (
                    not prediction
                    or prediction.scan_id != scan_id
                ):
                    raise ValueError(
                        "Prediction does not belong to this scan."
                    )

            review = DoctorReview(
                scan_id=scan_id,
                prediction_id=prediction_id,
                reviewer_id=reviewer_id,
                comments=comments,
                review_status=review_status,
            )

            session.add(review)
            session.flush()

            review_id = review.id

            session.add(
                AuditLog(
                    actor_id=reviewer_id,
                    action="DOCTOR_REVIEW_CREATED",
                    resource_type="DOCTOR_REVIEW",
                    resource_id=str(review_id),
                    details={
                        "scan_id": scan_id,
                        "review_status": review_status,
                    },
                )
            )

        print("Doctor review created. ID:", review_id)

        return review_id

    # --------------------------------------------------------
    # GET PATIENT HISTORY
    # --------------------------------------------------------

    def get_patient_history(self, patient_id):

        with self.session_factory() as session:

            patient = session.get(
                Patient,
                patient_id,
            )

            if not patient:
                raise ValueError("Patient not found.")

            result = {
                "patient_id": patient.id,
                "patient_code": patient.patient_code,
                "display_name": patient.display_name,
                "scans": [],
            }

            for scan in patient.scans:

                scan_data = {
                    "scan_id": scan.id,
                    "filename": scan.original_filename,
                    "modality": scan.modality,
                    "status": scan.status,
                    "predictions": [],
                    "reviews": [],
                }

                for prediction in scan.predictions:

                    scan_data["predictions"].append({
                        "prediction_id": prediction.id,
                        "predicted_class": prediction.predicted_class,
                        "confidence": prediction.confidence,
                        "probabilities": prediction.probabilities,
                        "model_version": prediction.model_version,
                    })

                for review in scan.reviews:

                    scan_data["reviews"].append({
                        "review_id": review.id,
                        "reviewer_id": review.reviewer_id,
                        "status": review.review_status,
                        "comments": review.comments,
                    })

                result["scans"].append(scan_data)

            return result


# ============================================================
# DATABASE STATISTICS
# ============================================================

def show_database_statistics():

    print("\n========== DATABASE STATISTICS ==========")

    with SessionLocal() as session:

        for model in [
            User,
            Patient,
            Scan,
            Prediction,
            DoctorReview,
            AuditLog,
        ]:

            count = session.scalar(
                select(func.count()).select_from(model)
            )

            print(
                f"{model.__tablename__:20s}: {count}"
            )


# ============================================================
# COMMAND LINE
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description="Phase 11 PostgreSQL Database Manager"
    )

    parser.add_argument(
        "--init",
        action="store_true",
        help="Create all database tables",
    )

    parser.add_argument(
        "--check",
        action="store_true",
        help="Test database connection",
    )

    parser.add_argument(
        "--stats",
        action="store_true",
        help="Display database table counts",
    )

    args = parser.parse_args()

    manager = DatabaseManager()

    try:

        if args.check:
            manager.check_connection()

        if args.init:
            manager.initialize()

        if args.stats:
            show_database_statistics()

        if not any([
            args.init,
            args.check,
            args.stats,
        ]):
            parser.print_help()

    except SQLAlchemyError as error:

        print("\nDATABASE ERROR:")
        print(error)
        sys.exit(1)

    except Exception as error:

        print("\nERROR:")
        print(error)
        sys.exit(1)


if __name__ == "__main__":
    main()
