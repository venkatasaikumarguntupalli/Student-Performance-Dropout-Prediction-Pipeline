import pandas as pd
import numpy as np

np.random.seed(42)
n = 500

attendance = np.random.randint(50, 100, n)
assignment_score = np.random.randint(40, 100, n)
gpa = np.round(np.random.uniform(1.5, 4.0, n), 2)
missed_submissions = np.random.randint(0, 10, n)
participation_score = np.random.randint(30, 100, n)

risk_score = (
    (100 - attendance) * 0.25
    + (100 - assignment_score) * 0.25
    + (4.0 - gpa) * 15
    + missed_submissions * 3
    + (100 - participation_score) * 0.15
)

dropout_risk = (risk_score > 35).astype(int)

df = pd.DataFrame({
    "attendance": attendance,
    "assignment_score": assignment_score,
    "gpa": gpa,
    "missed_submissions": missed_submissions,
    "participation_score": participation_score,
    "dropout_risk": dropout_risk
})

df.to_csv("data/student_data.csv", index=False)
print("Dataset created at data/student_data.csv")