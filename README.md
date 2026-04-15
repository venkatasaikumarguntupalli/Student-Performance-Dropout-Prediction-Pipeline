# Student-Performance-Dropout-Prediction-Pipeline

This project is a simple end-to-end machine learning workflow built to demonstrate **MLflow experiment tracking** and basic **MLOps concepts**.

The goal is to predict whether a student is at risk of dropping out based on features such as attendance, assignment performance, GPA, missed submissions, and participation score.

## Why this project

I built this project to get hands-on experience with:

- MLflow experiment tracking
- comparing multiple ML models
- logging metrics, parameters, and artifacts
- model selection and basic model lifecycle management

This project is intentionally simple, but it reflects the kind of structured experimentation workflow used in real-world ML systems.

## Tech Stack

- Python
- Pandas
- Scikit-learn
- Matplotlib
- MLflow

## Project Structure

```text
student-mlflow-project/
├── data/
│   └── student_data.csv
├── artifacts/
├── generate_data.py
├── train.py
├── requirements.txt
├── README.md
└── .gitignore
```

# Problem Statement

Educational institutions often want to identify students who may need support early.

This project predicts whether a student is at risk using a classification approach based on academic and engagement-related features.

# Features Used

Attendance
Assignment score
GPA
Missed submissions
Participation score

# Models Compared

The following models were trained and evaluated:

Logistic Regression
Decision Tree Classifier
Random Forest Classifier

# MLflow Usage

MLflow was used for:

tracking experiment runs
logging model parameters
logging evaluation metrics such as accuracy and F1-score
storing artifacts such as confusion matrix images
comparing runs to identify the best-performing model

# setup instructions

1. Clone the repository

```bash
git clone https://github.com/venkatasaikumarguntupalli/k8s-log-aggregator.git
cd student-mlflow-project
```

2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate
```

3. Install dependencies

```bash
pip install -r requirements.txt
```

# Run Project

Generate sample data

```bash
python generate_data.py
```

This creates:

```bash
data/student_data.csv
```

# Train models and log experiments

```bash
data/student_data.csv
```

This will:

train multiple models
evaluate each model
create confusion matrix artifacts
log metrics and parameters to MLflow

# Launch MLflow UI

```bash
mlflow ui
```

Then open:

```bash
http://127.0.0.1:5000
```

# Sample Outputs

After running the project, the following outputs are generated:

MLflow experiment runs
logged parameters and metrics
confusion matrix images for each model

Example artifact files:

logistic_regression_confusion_matrix.png
decision_tree_confusion_matrix.png
random_forest_confusion_matrix.png

# Interpreting the Results

The confusion matrices help compare how well each model predicts student risk.

For example:

Logistic Regression showed very strong performance on the generated dataset
Decision Tree performed reasonably well but made more classification errors
Random Forest also performed strongly, especially on positive class detection

The exact best model may vary depending on the generated dataset and train/test split.

# What I Learned

Through this project, I learned how MLflow helps organize machine learning experimentation by making it easier to:

compare different models
reproduce runs
track training decisions
keep artifacts and metrics in one place

# Possible improvements

Future enhancements could include:

hyperparameter tuning
model registry integration
Dockerizing the project
deploying the workflow on Kubernetes or OpenShift
exposing predictions through a FastAPI endpoint
using a real dataset instead of generated synthetic data
