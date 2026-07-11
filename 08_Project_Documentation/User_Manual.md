# OptiCrop - User Manual

This manual explains how to access and use the **OptiCrop – Intelligent Crop Recommendation System**.

---

## 1. Accessing the Application

To run the OptiCrop application:

1. Open the project in VS Code.
2. Navigate to the `06_Web_Application` folder.
3. Run the Flask application using:

```bash
python app.py
```

4. Open a web browser and enter:

```
http://127.0.0.1:5000
```

5. The OptiCrop Home Page will be displayed.

---

## 2. Entering Input Values

The application requires seven input parameters.

### Soil Parameters

1. **Nitrogen (N)** – Enter the Nitrogen value.
2. **Phosphorus (P)** – Enter the Phosphorus value.
3. **Potassium (K)** – Enter the Potassium value.
4. **Soil pH** – Enter the soil pH value.

### Environmental Parameters

5. **Temperature (°C)** – Enter the temperature.
6. **Humidity (%)** – Enter the humidity.
7. **Rainfall (mm)** – Enter the rainfall.

Ensure all values are entered correctly before making a prediction.

---

## 3. Predicting a Crop

1. Enter all required input values.
2. Click the **Predict Crop** button.
3. The application processes the input using the trained Machine Learning model.
4. The Result Page displays the recommended crop.

---

## 4. Understanding the Prediction Result

The Result Page displays:

### Recommended Crop

The application displays the most suitable crop based on the entered soil and climate parameters.

### Input Summary

The entered values are shown for verification.

### Predict Again

Click the **Predict Again** button to return to the Home Page and perform another prediction.

---

## 5. Troubleshooting Common Issues

### Issue 1: Website does not open

**Cause:** Flask server is not running.

**Solution:**

Run:

```bash
python app.py
```

Then open:

```
http://127.0.0.1:5000
```

---

### Issue 2: Prediction is not displayed

**Cause:** Invalid or incomplete input values.

**Solution:**

Check that all fields contain valid numerical values within the specified range.

---

### Issue 3: Model file not found

**Cause:** `crop_model.pkl` is missing.

**Solution:**

Place the `crop_model.pkl` file inside the **model** folder and restart the application.

---

### Issue 4: Application Error

**Cause:** Required Python packages are not installed.

**Solution:**

Install the required dependencies using:

```bash
pip install -r requirements.txt
```

Restart the Flask application after installation.