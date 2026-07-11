# 🌿 OptiCrop - Intelligent Crop Recommendation System

OptiCrop is a Machine Learning-based web application that recommends the most suitable crop based on soil nutrients and environmental conditions. The system helps farmers and agriculture enthusiasts make informed crop selection decisions using data-driven predictions.

---

# Project Objectives

- Recommend suitable crops using Machine Learning.
- Improve agricultural productivity through data-driven decisions.
- Assist farmers in selecting crops based on soil and climate conditions.
- Provide a simple and user-friendly web application.

---

# Features

- Machine Learning-based Crop Recommendation
- Flask Web Application
- Responsive User Interface
- Soil and Climate Parameter Input
- Instant Crop Prediction
- Input Validation
- Example Input Values for Testing

---

# Technology Stack

## Backend

- Python
- Flask
- Scikit-learn
- Pandas
- NumPy
- Joblib

## Frontend

- HTML5
- CSS3
- Bootstrap 5
- JavaScript

---

# Folder Structure

```text
OptiCrop/
│
├── 01_Problem_Definition/
├── 02_Requirement_Analysis/
├── 03_Data_Analysis/
├── 04_Preprocessing/
├── 05_Model_Building/
├── 06_Web_Application/
├── 07_Testing/
├── 08_Project_Documentation/
├── 09_Project_Demonstration/
│
├── Dataset/
│   └── Crop_recommendation.csv
│
└── README.md
```

---

# Application Workflow

```text
User
   ↓
Open OptiCrop Website
   ↓
Enter Soil & Climate Parameters
   ↓
Input Validation
   ↓
Click Predict Crop
   ↓
Flask Application
   ↓
Load crop_model.pkl
   ↓
Machine Learning Prediction
   ↓
Display Recommended Crop
```

---

# Installation & Running the Project

## Prerequisites

- Python 3.8 or above
- Git (Optional)

## Steps

### Clone the repository

```bash
git clone https://github.com/vaibhav413/OptiCrop.git
```

### Open the project

```bash
cd OptiCrop
```

### Install required packages

```bash
pip install -r 06_Web_Application/requirements.txt
```

### Run the application

```bash
cd 06_Web_Application
python app.py
```

### Open in Browser

```
http://127.0.0.1:5000
```

---

# Dataset

The Machine Learning model was trained using the **Crop_recommendation.csv** dataset.

The dataset contains the following features:

- Nitrogen (N)
- Phosphorus (P)
- Potassium (K)
- Temperature
- Humidity
- Soil pH
- Rainfall
- Crop Label

---

# Sample Input

| Parameter | Value |
|-----------|------:|
| Nitrogen | 90 |
| Phosphorus | 42 |
| Potassium | 43 |
| Temperature | 21°C |
| Humidity | 82% |
| Soil pH | 6.5 |
| Rainfall | 200 mm |

---

# Sample Output

**Recommended Crop**

```
Rice
```

---

# Future Scope

- Fertilizer Recommendation
- Crop Disease Detection
- Weather API Integration
- Yield Prediction
- User Login and Prediction History

---

# Developer

**YEGGEPALLI SAI SRINIVASA VAIBHAV**

**GitHub:** vaibhav413

**College:** Vishnu Institute of Technology

**Department:** Computer Science

**Year:** 4th Year

---

# License

This project was developed for academic and educational purposes as a Final Year Machine Learning Project.