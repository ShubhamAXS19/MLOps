# src/models/train_model.py
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
import joblib
import os

def build_and_train_model(X_train, y_train):
    """
    Build and train a Random Forest model.
    """
    model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_val, y_val):
    """
    Evaluate the model on the validation set and log metrics with MLflow.
    """
    y_pred = model.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)
    precision = precision_score(y_val, y_pred)
    recall = recall_score(y_val, y_pred)
    auc = roc_auc_score(y_val, y_pred)

    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("auc", auc)

    return accuracy, precision, recall, auc

def save_model(model, model_dir="models/"):
    """
    Save the trained model to the specified directory.
    """
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "random_forest_model.pkl")
    joblib.dump(model, model_path)
    mlflow.log_artifact(model_path)
    print(f"Model saved to {model_path}")
    return model_path

if __name__ == "__main__":
    # Assuming the data has been preprocessed and is available in the following variables
    from src.features.preprocess_data import preprocess_data

    data_dir = "data/raw"
    X_train, X_val, X_test, y_train, y_val, y_test = preprocess_data(data_dir)
    
    with mlflow.start_run():
        model = build_and_train_model(X_train, y_train)
        evaluate_model(model, X_val, y_val)
        save_model(model)
