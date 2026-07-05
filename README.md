# YouTube Adview Prediction System

A machine learning project that analyzes YouTube video engagement data to estimate advertisement views and classify expected video performance.

The project includes the complete workflow from data preprocessing and feature engineering to model training and prediction through an interactive Streamlit web application called **AdVision AI**.

---

## Project Overview

YouTube video performance is influenced by multiple factors such as views, likes, dislikes, comments, duration, category, and publishing information. Analyzing these factors manually can be difficult when working with large datasets.

The **YouTube Adview Prediction System** uses machine learning to process video-level information and generate two outputs:

- **Predicted Adviews** — estimates the expected number of advertisement views
- **Performance Segment** — classifies the video into an expected performance category

The project demonstrates how a trained machine learning pipeline can be integrated into a practical web application for interactive prediction.

---

## Application Preview

The Streamlit application is presented under the name **AdVision AI** and provides a structured interface for prediction and model exploration.

Main sections include:

- **Overview** — summary of the platform and machine learning workflow
- **Predict** — accepts video metrics and generates predictions
- **Model Insights** — presents model and feature information
- **How It Works** — explains the prediction pipeline
- **About Project** — provides technical project details

---

## Key Features

- Predicts estimated YouTube advertisement views
- Classifies expected video performance
- Accepts video engagement and publishing metrics as input
- Automatically generates derived engagement features
- Maintains the same feature structure used during model training
- Uses saved preprocessing objects for consistent inference
- Provides real-time predictions through Streamlit
- Includes a responsive multi-page web interface
- Separates regression and classification tasks within one application

---

## Machine Learning Tasks

### Adview Prediction

The regression component estimates advertisement views based on the processed video features.

**Model used:** Random Forest Regressor

**Output:** Estimated advertisement views

### Performance Classification

The classification component assigns the video to an expected performance segment.

**Model used:** Random Forest Classifier

**Output categories:**

- Low Performance
- Average Performance
- High Performance
- Viral Performance

---

## Input Features

The application collects the following information from the user:

| Feature | Description |
|---|---|
| Category | YouTube video category |
| Views | Total number of video views |
| Likes | Total number of likes |
| Dislikes | Total number of dislikes |
| Comments | Total number of comments |
| Duration | Video duration in seconds |
| Published Year | Year in which the video was published |
| Published Month | Month in which the video was published |

The application then calculates additional derived features:

| Derived Feature | Description |
|---|---|
| Engagement | Combined likes and comments |
| Like Rate | Likes relative to total views |
| Comment Rate | Comments relative to total views |
| Dislike Rate | Dislikes relative to total views |

This results in **12 predictive features** used by the application pipeline.

---

## Prediction Workflow

```text
User Input
    │
    ▼
Input Validation
    │
    ▼
Category Encoding
    │
    ▼
Feature Engineering
    │
    ▼
Feature Alignment
    │
    ├─────────────────────┐
    ▼                     ▼
Regression Model     Classification Model
    │                     │
    ▼                     ▼
Predicted Adviews    Performance Segment
    │                     │
    └──────────┬──────────┘
               ▼
        Streamlit Results
```

---

## Technology Stack

| Area | Technologies |
|---|---|
| Programming | Python |
| Machine Learning | Scikit-learn |
| Data Processing | Pandas, NumPy |
| Regression | Random Forest Regressor |
| Classification | Random Forest Classifier |
| Web Application | Streamlit |
| Interface Styling | HTML, CSS |
| Development | Jupyter Notebook, Anaconda |
| Version Control | Git, GitHub |

---

## Project Structure

```text
YouTube-Adview-Prediction-System/
│
├── app.py
├── YouTube Adview Prediction System.ipynb
├── category_encoder.pkl
├── feature_columns.pkl
├── .gitignore
└── README.md
```

### File Description

| File | Purpose |
|---|---|
| `app.py` | Main Streamlit web application |
| `YouTube Adview Prediction System.ipynb` | Data processing, model development, and experimentation notebook |
| `category_encoder.pkl` | Saved encoder used for video category transformation |
| `feature_columns.pkl` | Saved feature schema used to maintain prediction input order |
| `.gitignore` | Excludes files that should not be committed to the repository |
| `README.md` | Project documentation |

---

## Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/Rakshithavr5/YouTube-Adview-Prediction-System.git
```

### 2. Open the Project Directory

```bash
cd YouTube-Adview-Prediction-System
```

### 3. Install Dependencies

```bash
pip install streamlit pandas numpy scikit-learn
```

### 4. Add the Trained Models

The application requires these trained model files:

```text
adview_regression_model.pkl
adview_segment_classifier.pkl
```

Place both files in the project root directory alongside `app.py`.

### 5. Run the Application

```bash
streamlit run app.py
```

After the Streamlit server starts, open the local address displayed in the terminal.

---

## Model Files
The trained regression and classification models are not stored in this repository because both exceed GitHub's standard **100 MB per-file limit**.

| Model File | Purpose |
|---|---|
| `adview_regression_model.pkl` | Generates estimated advertisement views |
| `adview_segment_classifier.pkl` | Predicts the expected performance segment |

These files are excluded from Git tracking through `.gitignore`.

> **Important:** The complete prediction application requires both model files to be available locally in the project root directory.

---
## Running the Prediction System

After launching the application:

1. Open the **Predict** section from the sidebar
2. Select the video category
3. Enter video engagement statistics
4. Provide duration and publishing information
5. Click **Generate Prediction**
6. View the estimated adviews and predicted performance segment

---
## Current Limitations

- The trained model files are not included directly in the repository because of GitHub file-size restrictions
- Prediction quality depends on the dataset and feature patterns used during model training
- The current application expects the same feature schema used during training
- Serialized scikit-learn models should be loaded with a compatible library version

---

## Author

**Rakshitha V R**

B.Tech Information Technology  
Interested in Software Development, Machine Learning, and Data-Driven Applications

---

## Repository

Project source code and development files are maintained in this repository.

If you find the project useful, consider giving it a star.
