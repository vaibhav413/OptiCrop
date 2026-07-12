# 🌱 OptiCrop

<div align="center">

## Intelligent Crop Recommendation System using Machine Learning

### Individual Project

Developed by

# YEGGEPALLI SAI SRINIVASA VAIBHAV

**B.Tech – Computer Science & Engineering**

**Vishnu Institute of Technology**

**4th Year**

**GitHub:** https://github.com/vaibhav413


**website:** https://opticrop-mius.onrender.com/

</div>

---

# 📖 Project Overview

OptiCrop is an intelligent Crop Recommendation System developed using **Machine Learning** and **Flask**. The application predicts the most suitable crop based on soil nutrient levels and environmental conditions, helping users make informed agricultural decisions.

The project demonstrates the complete Machine Learning lifecycle—from data preprocessing and model training to deployment through an interactive Flask web application.

OptiCrop features a responsive user interface, real-time prediction, input validation, and confidence score visualization, making it easy to use for learning and demonstration purposes.

---

# 🎯 Objectives

- Develop an intelligent crop recommendation system.
- Predict the most suitable crop using soil and climate parameters.
- Build a responsive web application using Flask.
- Deploy a trained Machine Learning model for real-time predictions.
- Demonstrate an end-to-end Machine Learning workflow.

---

# ✨ Features

- 🌱 AI-Based Crop Recommendation
- 🤖 Machine Learning Prediction Model
- 🎨 Modern Responsive User Interface
- 📊 Confidence Score Display
- ✅ Input Validation
- 📱 Mobile-Friendly Design
- ⚡ Fast Prediction
- 🌾 Sample Input Values
- 🌍 User-Friendly Dashboard

---

# 🛠 Technologies Used

## Programming Languages

- Python
- HTML5
- CSS3
- JavaScript

## Framework

- Flask

## Machine Learning Libraries

- Scikit-Learn
- Pandas
- NumPy
- Joblib

## Frontend

- Bootstrap 5
- Bootstrap Icons
- Google Fonts

## Development Tools

- Visual Studio Code
- Git
- GitHub
---

# 📊 Dataset Used

The Machine Learning model used in **OptiCrop** was trained using the **Crop Recommendation Dataset (`Crop_recommendation.csv`)**, which is included in this repository.

This dataset contains soil nutrient values and environmental parameters that are used to recommend the most suitable crop for cultivation.

### Dataset Features

| Feature | Description |
|----------|-------------|
| Nitrogen (N) | Nitrogen content in the soil |
| Phosphorus (P) | Phosphorus content in the soil |
| Potassium (K) | Potassium content in the soil |
| Temperature | Atmospheric temperature (°C) |
| Humidity | Relative humidity (%) |
| Soil pH | Soil acidity/alkalinity |
| Rainfall | Annual rainfall (mm) |

### Target Variable

- Crop Label (Recommended Crop)

### Dataset Location

```
Dataset/
└── Crop_recommendation.csv
```

The dataset was preprocessed, analyzed, and used to train the Machine Learning model before deploying it through the Flask web application.

---

# 📂 Project Structure

```
OptiCrop
│
├── 01_Project_Overview
├── 02_Dataset
├── 03_Data_Preprocessing
├── 04_Model_Training
├── 05_Model_Evaluation
├── 06_Web_Application
│   │
│   ├── app.py
│   ├── requirements.txt
│   │
│   ├── model
│   │     └── crop_model.pkl
│   │
│   ├── static
│   │     ├── css
│   │     ├── js
│   │     └── images
│   │
│   └── templates
│         ├── index.html
│         └── result.html
│
├── README.md
├── LICENSE
└── .gitignore
```

---

# 🚀 Installation

### Step 1: Clone the Repository

```bash
git clone https://github.com/vaibhav413/OptiCrop.git
```

---

### Step 2: Navigate to the Project Folder

```bash
cd OptiCrop
```

---

### Step 3: Install the Required Dependencies

```bash
pip install -r requirements.txt
```

---

### Step 4: Launch the Flask Application

```bash
python app.py
```

---

### Step 5: Open the Application

Open your browser and visit:

```
http://127.0.0.1:5000
```

---

# 💻 Input Parameters

The prediction model uses the following seven input parameters:

| Parameter | Valid Range |
|-----------|-------------|
| Nitrogen (N) | 0 – 200 mg/kg |
| Phosphorus (P) | 0 – 200 mg/kg |
| Potassium (K) | 0 – 300 mg/kg |
| Temperature | -10°C to 60°C |
| Humidity | 0 – 100% |
| Soil pH | 0 – 14 |
| Rainfall | 0 – 500 mm |

After entering the required values, click **Predict Crop** to receive the recommended crop along with the model's confidence score.
---

# 🌟 Key Features

- 🌱 Intelligent Crop Recommendation using Machine Learning
- 🤖 Predicts the most suitable crop based on soil and environmental conditions
- 📊 Displays prediction confidence score
- 🎨 Modern, responsive, and user-friendly interface
- ✅ Input validation for reliable predictions
- 📱 Mobile-friendly design built with Bootstrap 5
- ⚡ Fast prediction using a pre-trained Machine Learning model
- 🌾 Example input values for quick testing
- 💻 Flask-based web application
- 🔒 Clean and organized project structure

---

# 🔮 Future Enhancements

The following features can be added to further improve the project:

- 🌦 Live Weather API Integration
- 💧 Fertilizer Recommendation System
- 🚜 Smart Irrigation Recommendation
- 📈 Crop Yield Prediction
- 🌍 Multi-language Support
- 📊 Farmer Analytics Dashboard
- ☁ Cloud Deployment
- 📱 Android Mobile Application
- 🛰 IoT-based Soil Sensor Integration

---

# 🎯 Learning Outcomes

Through the development of **OptiCrop**, I gained practical experience in:

- Machine Learning model development
- Data preprocessing and feature engineering
- Exploratory Data Analysis (EDA)
- Model training and evaluation
- Flask web application development
- Frontend development using HTML, CSS, JavaScript, and Bootstrap
- Deploying Machine Learning models in web applications
- Git and GitHub version control
- Building complete end-to-end Machine Learning projects

---

# 👨‍💻 Author

## YEGGEPALLI SAI SRINIVASA VAIBHAV

**B.Tech – Computer Science & Engineering**

**Vishnu Institute of Technology**

**4th Year**

### GitHub

🔗 https://github.com/vaibhav413

---

# 🤝 Acknowledgements

This is an **individual academic project** developed to demonstrate the practical application of **Machine Learning** in agriculture.

The Machine Learning model was trained using the **Crop Recommendation Dataset (`Crop_recommendation.csv`)**, and the project demonstrates the complete workflow including:

- Dataset preprocessing
- Exploratory Data Analysis
- Model training
- Model evaluation
- Flask web application deployment

---

# 📄 License

This project is developed for **educational and learning purposes**.

---

<div align="center">

## ⭐ Thank you for visiting my repository!

If you found this project interesting or useful, please consider giving it a ⭐ on GitHub.

### 🌱 Made with Python, Flask, Bootstrap, and Machine Learning

</div>
