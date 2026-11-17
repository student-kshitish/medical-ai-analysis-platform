# Medical AI Platform - Project Requirements

## 1. Project Overview

An explainable AI-powered medical imaging platform
that analyzes medical images and assists doctors
through deep learning and visual explanations.

## 2. Initial Scope

Initial modality: Chest X-ray
Initial task: Pneumonia classification

Future modalities:
- CT
- MRI

## 3. User Roles

### Patient/User
- Register and login
- Upload medical images
- View analysis results
- View analysis history
- View doctor review status

### Doctor
- Login
- View assigned scans
- View original medical images
- View AI predictions
- View Grad-CAM heatmaps
- Add comments
- Submit review

## 4. AI Requirements

- DICOM image reading
- Image preprocessing
- MONAI integration
- PyTorch model training
- Model inference
- Prediction probabilities
- Grad-CAM explainability
- Model evaluation

## 5. Application Requirements

- React frontend
- FastAPI backend
- PostgreSQL database
- Orthanc DICOM server
- Docker deployment
- Authentication and authorization

## 6. Evaluation Requirements

- Accuracy
- Precision
- Recall
- Sensitivity
- Specificity
- F1 score
- AUROC
- Confusion matrix

## 7. Testing Requirements

- Unit tests
- API tests
- Model tests
- DICOM tests
- Integration tests
- End-to-end tests

## 8. Safety and Limitations

This project is an educational research prototype.
AI predictions are not medical diagnoses.
Doctor review and independent clinical validation
are required before any clinical application.
