# OptiCrop - Testing Module (Epic 6)

This directory contains the testing documents, test cases, and test reports for the **OptiCrop – Intelligent Crop Recommendation System**.

The purpose of this module is to verify that the Machine Learning model, Flask web application, and user interface work correctly and provide accurate crop recommendations.

---

## Testing Workflow

```text
Start Application
        │
        ▼
Run Test Cases
        │
        ▼
Verify Dataset
        │
        ▼
Verify Model Loading
        │
        ▼
Verify Crop Prediction
        │
        ▼
Verify Flask Application
        │
        ▼
Verify Form Validation
        │
        ▼
Verify User Interface
        │
        ▼
Generate Test Report
```

---

## Modules Tested

1. **Dataset:** Verified that `Crop_recommendation.csv` is available and readable.
2. **Model:** Verified successful loading of `crop_model.pkl`.
3. **Prediction:** Verified accurate crop recommendations using sample inputs.
4. **Flask Application:** Tested Home Page and Result Page functionality.
5. **Form Validation:** Verified valid and invalid user inputs.
6. **User Interface:** Verified responsive layout and navigation.
7. **Performance:** Verified prediction response time.

---

## Folder Structure

```text
07_Testing/
│
├── README.md              # Testing documentation
├── testing.py             # Test script
├── test_cases.md          # Test cases
├── test_report.md         # Test report
├── manual_testing.md      # Manual testing guide
│
├── outputs/
│   └── test_summary.txt   # Test summary
│
└── screenshots/
    ├── home_page.png
    ├── prediction_page.png
    ├── result_page.png
    └── validation.png
```

---

## Test Cases Summary

The testing module verifies:

- Dataset availability
- Model loading
- Crop prediction
- Flask application
- Form validation
- User interface
- Performance

Refer to **test_cases.md** for detailed test cases.

---

## How to Execute Testing

### 1. Run Test Script

Open the terminal and execute:

```bash
python testing.py
```

or

```bash
cd 07_Testing
python testing.py
```

The test summary will be generated in:

```
07_Testing/outputs/test_summary.txt
```

---

### 2. Manual Testing

Refer to **manual_testing.md** for step-by-step instructions to test the application manually.

---

## Future Improvements

- Add automated UI testing using Selenium or Playwright.
- Integrate testing with GitHub Actions.
- Improve performance testing for larger workloads.
- Add more validation test cases for user inputs.