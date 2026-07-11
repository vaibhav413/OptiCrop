# OptiCrop - Manual Testing Guide

This document provides step-by-step instructions for manually testing the **OptiCrop – Intelligent Crop Recommendation System**.

---

# 1. Home Page Testing

## Objective

Verify that the home page loads correctly and displays all required components.

## Steps

1. Open a web browser and navigate to:

   `http://127.0.0.1:5000`

2. Verify that the page title is **OptiCrop | Smart Crop Recommendation**.

3. Verify that the home page displays:

   - Navigation Bar
   - Project Title
   - Hero Image
   - Crop Recommendation Form
   - Example Input Values
   - About Section
   - Footer

4. Verify that the form contains the following input fields:

   - Nitrogen (N)
   - Phosphorus (P)
   - Potassium (K)
   - Temperature
   - Humidity
   - Soil pH
   - Rainfall

5. Verify that the **Predict Crop** button is displayed.

---

# 2. Prediction Testing

## Objective

Verify that the application predicts the correct crop.

## Steps

1. Open the OptiCrop home page.

2. Enter the following sample values:

   - Nitrogen = 90
   - Phosphorus = 42
   - Potassium = 43
   - Temperature = 21
   - Humidity = 82
   - Soil pH = 6.5
   - Rainfall = 200

3. Click **Predict Crop**.

4. Verify that the application redirects to the Result Page.

5. Verify that the following are displayed:

   - Recommended Crop
   - Prediction Confidence Score
   - Soil & Climate Summary
   - Predict Again Button

---

# 3. Form Validation Testing

## Objective

Verify that the form accepts only valid inputs.

## Steps

### Empty Fields

1. Leave one or more fields empty.

2. Click **Predict Crop**.

3. Verify that the browser prevents form submission.

---

### Invalid Values

1. Enter values outside the specified range.

2. Verify that validation prevents invalid input.

---

### Alphabetic Input

1. Try entering letters into the numeric fields.

2. Verify that only numeric values are accepted.

---

# 4. Navigation Testing

## Objective

Verify that navigation works correctly.

## Steps

1. Open the home page.

2. Click **Try Prediction**.

3. Verify that the page scrolls to the prediction form.

4. Perform a prediction.

5. Click **Predict Again**.

6. Verify that the application returns to the Home Page.

---

# 5. Responsive Design Testing

## Objective

Verify that the application works properly on different screen sizes.

## Steps

1. Open the website.

2. Resize the browser window or use Developer Tools (F12).

3. Verify that:

- The layout adjusts correctly.
- Input fields remain visible.
- Buttons are accessible.
- Images scale properly.
- No text overlaps.

---

# 6. Error Handling Testing

## Objective

Verify that the application handles errors properly.

## Steps

1. Temporarily rename:

```
crop_model.pkl
```

to

```
crop_model_temp.pkl
```

2. Run the application.

3. Try making a prediction.

4. Verify that the application displays an appropriate error message instead of crashing.

5. Rename the model back to:

```
crop_model.pkl
```

6. Verify that prediction works normally.