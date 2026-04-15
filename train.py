import os
import mlflow
import mlflow.sklearn
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier


def load_data():
    df = pd.read_csv("data/student_data.csv")
    X = df.drop("dropout_risk", axis=1)
    y = df["dropout_risk"]
    return train_test_split(X, y, test_size=0.2, random_state=42)


def plot_confusion_matrix(y_true, y_pred, model_name):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    os.makedirs("artifacts", exist_ok=True)
    file_path = f"artifacts/{model_name}_confusion_matrix.png"
    plt.savefig(file_path)
    plt.close()
    return file_path


def train_and_log_model(model, model_name, X_train, X_test, y_train, y_test):
    with mlflow.start_run(run_name=model_name):
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        mlflow.log_param("model_name", model_name)

        if hasattr(model, "get_params"):
            for key, value in model.get_params().items():
                mlflow.log_param(key, value)

        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("f1_score", f1)

        cm_path = plot_confusion_matrix(y_test, y_pred, model_name)
        mlflow.log_artifact(cm_path)

        mlflow.sklearn.log_model(model, name="model")

        print(f"{model_name} -> Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
        return accuracy, f1


def main():
    mlflow.set_experiment("student-risk-prediction")

    X_train, X_test, y_train, y_test = load_data()

    models = {
        "logistic_regression": LogisticRegression(max_iter=300),
        "decision_tree": DecisionTreeClassifier(max_depth=5, random_state=42),
        "random_forest": RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42),
    }

    best_model_name = None
    best_f1 = -1
    best_model = None

    for model_name, model in models.items():
        accuracy, f1 = train_and_log_model(model, model_name, X_train, X_test, y_train, y_test)
        if f1 > best_f1:
            best_f1 = f1
            best_model_name = model_name
            best_model = model

    with mlflow.start_run(run_name="best_model_registration"):
        mlflow.log_param("best_model_name", best_model_name)
        mlflow.log_metric("best_f1_score", best_f1)

        model_uri = f"runs:/{mlflow.active_run().info.run_id}/best_model_placeholder"
        print(f"Best model selected: {best_model_name} with F1 score {best_f1:.4f}")

    print("Training complete.")


if __name__ == "__main__":
    main()