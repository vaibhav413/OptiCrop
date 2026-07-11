# OptiCrop Test Cases

This document lists the test cases used to verify the functionality and performance of the **OptiCrop – Intelligent Crop Recommendation System**.

| Test Case ID | Module | Input / Action | Expected Output | Actual Output | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **TC-001** | Dataset Verification | Check whether `Crop_recommendation.csv` exists. | Dataset file is available. | Dataset found successfully. | **Pass** |
| **TC-002** | Dataset Verification | Load dataset using Pandas. | Dataset loads without errors. | Dataset loaded successfully. | **Pass** |
| **TC-003** | Dataset Verification | Verify required columns. | All required features are present. | All columns verified. | **Pass** |
| **TC-004** | Model Verification | Check whether `crop_model.pkl` exists. | Model file is available. | Model found successfully. | **Pass** |
| **TC-005** | Model Verification | Load model using Joblib. | Model loads successfully. | Model loaded successfully. | **Pass** |
| **TC-006** | Prediction Module | Enter valid soil and climate values. | Recommended crop is displayed. | Crop recommendation displayed successfully. | **Pass** |
| **TC-007** | Home Page | Open the application. | Home page loads correctly. | Home page displayed successfully. | **Pass** |
| **TC-008** | Result Page | Click **Predict Crop**. | Result page displays prediction. | Result displayed successfully. | **Pass** |
| **TC-009** | Form Validation | Leave one or more fields empty. | Form prevents submission. | Validation successful. | **Pass** |
| **TC-010** | Form Validation | Enter values outside the allowed range. | Invalid input is rejected. | Validation successful. | **Pass** |
| **TC-011** | Form Validation | Enter alphabetic characters in numeric fields. | Only numeric values are accepted. | Validation successful. | **Pass** |
| **TC-012** | Navigation | Click **Try Prediction**. | Page scrolls to prediction form. | Navigation successful. | **Pass** |
| **TC-013** | Navigation | Click **Predict Again** after prediction. | Returns to Home Page. | Navigation successful. | **Pass** |
| **TC-014** | Responsive Design | Open website on different screen sizes. | Layout adjusts properly. | Responsive layout verified. | **Pass** |
| **TC-015** | Performance | Submit valid input for prediction. | Prediction generated within acceptable time. | Prediction completed successfully. | **Pass** |