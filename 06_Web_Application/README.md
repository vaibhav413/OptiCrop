# OptiCrop - Web Application (Epic 5)

This directory contains the complete web application for **OptiCrop – Intelligent Crop Recommendation System**. The application is developed using **Flask (Python)** for the backend and **HTML, CSS, Bootstrap, and JavaScript** for the frontend.

The application uses a trained Machine Learning model (`crop_model.pkl`) to recommend the most suitable crop based on soil nutrient values and environmental conditions.

---

## Technologies Used

* **Backend:** Flask (Python), Pandas, NumPy, Scikit-learn, Joblib
* **Frontend:** HTML5, CSS3, Bootstrap 5, JavaScript
* **Model:** Trained Crop Recommendation Model (`crop_model.pkl`)

---

## Folder Structure

```text
06_Web_Application/
│
├── app.py                  # Main Flask application
├── requirements.txt        # Python dependencies
├── README.md               # Documentation
│
├── model/
│   └── crop_model.pkl      # Trained ML model
│
├── templates/
│   ├── index.html          # Home page
│   └── result.html         # Prediction result page
│
├── static/
│   ├── css/
│   │   └── style.css       # Stylesheet
│   ├── js/
│   │   └── script.js       # JavaScript
│   └── images/             # Images used in the website
│
└── screenshots/            # Application screenshots
```

---

## Application Workflow

```text
User
   ↓
Open OptiCrop Website
   ↓
Enter Soil & Climate Parameters
   ↓
Click "Predict Crop"
   ↓
Flask Receives Input
   ↓
Load crop_model.pkl
   ↓
Machine Learning Prediction
   ↓
Display Recommended Crop
```

---

## Installation Steps

### 1. Prerequisite

Install **Python 3.8 or above**.

### 2. Create Virtual Environment (Optional)

```bash
python -m venv .venv
```

Activate the virtual environment:

**Windows**

```bash
.venv\Scripts\activate
```

**Linux/macOS**

```bash
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
cd 06_Web_Application
pip install -r requirements.txt
```

---

## How to Run

Run the Flask application using:

```bash
python app.py
```

Open your browser and visit:

```
http://127.0.0.1:5000
```

---

## Screenshots

- Home Page
- Crop Recommendation Form
- Result Page

---

## Future Scope

1. Fertilizer Recommendation
2. Crop Disease Detection
3. Weather API Integration
4. Yield Prediction
5. User Dashboard and Prediction History