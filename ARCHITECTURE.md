# System Architecture

## AIML-01: Medical Imaging Intelligence

Responsible for:
- Dataset management
- DICOM processing
- Image preprocessing
- MONAI integration
- PyTorch model training
- Model evaluation
- Grad-CAM
- AI inference

## AIML-02: Medical Imaging Application

Responsible for:
- React frontend
- FastAPI backend
- Orthanc integration
- PostgreSQL database
- User authentication
- Doctor review
- Application deployment

## System Flow

React Frontend
      |
      v
FastAPI Backend
      |
      +------ PostgreSQL
      |
      +------ Orthanc
      |
      +------ AI Inference Engine
                    |
                    +---- PyTorch
                    |
                    +---- MONAI
                    |
                    +---- Grad-CAM
      |
      v
Doctor Review

## Initial Modality

Chest X-ray

## Future Expansion

CT and MRI using modality-specific
preprocessing and deep learning models.
