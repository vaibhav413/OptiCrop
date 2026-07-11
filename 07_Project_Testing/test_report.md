# OptiCrop Test Report

## 1. Executive Summary

This report summarizes the testing phase (Epic 6) for the **OptiCrop – Intelligent Crop Recommendation System**. Testing was carried out to verify the dataset, Machine Learning model, Flask web application, prediction functionality, form validation, and user interface.

---

## 2. Test Execution Summary

| Metric | Value |
| :--- | :--- |
| **Total Test Cases** | 15 |
| **Passed** | 15 |
| **Failed** | 0 |
| **Pass Percentage** | **100%** |
| **System Status** | **Ready for Deployment** |

---

## 3. Detailed Observations

### A. Dataset & Model

- The dataset **Crop_recommendation.csv** was successfully loaded and verified.
- All required input features (Nitrogen, Phosphorus, Potassium, Temperature, Humidity, pH, and Rainfall) were present.
- The trained Machine Learning model **crop_model.pkl** loaded successfully.
- The model generated crop recommendations correctly for valid input values.

### B. Web Application

- The Home Page loaded successfully.
- The Crop Recommendation Form accepted valid user inputs.
- The Result Page displayed the recommended crop correctly.
- Navigation between the Home Page and Result Page worked properly.
- Form validation prevented invalid or empty inputs.

### C. User Interface

- The application displayed correctly on desktop and mobile devices.
- Buttons, forms, and images were responsive.
- The user interface was simple, clean, and easy to use.

### D. Performance

- The application loaded successfully.
- Crop predictions were generated quickly.
- No significant delays were observed during testing.

---

## 4. Recommendations for Future Work

1. Add fertilizer recommendation based on soil nutrients.
2. Integrate weather APIs for real-time environmental data.
3. Implement crop disease detection using image processing.
4. Store prediction history using a database.
5. Deploy the application on a cloud platform for public access.