# src/evaluate_model.py

import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pickle
import yaml

def load_params():
    with open("params.yaml", "r") as f:
        return yaml.safe_load(f)

def evaluate_model():
    params = load_params()

    # Load the test data
    test_data_path = params['data']['paths']['test_data']
    test_data = pd.read_csv(test_data_path)

    X_test = test_data.drop('target', axis=1)
    y_test = test_data['target']

    # Load the trained model
    with open("models/model.pkl", "rb") as f:
        model = pickle.load(f)

    # Load the scaler
    with open("models/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    
    # Scale the test data
    X_test_scaled = scaler.transform(X_test)

    # Make predictions
    y_pred = model.predict(X_test_scaled)

    # Calculate evaluation metrics
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average='macro'),
        "recall": recall_score(y_test, y_pred, average='macro'),
        "f1_score": f1_score(y_test, y_pred, average='macro')
    }

    print("Model Evaluation Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

    # Save the metrics to a file
    with open("metrics.json", "w") as f:
        yaml.dump(metrics, f)

if __name__ == "__main__":
    evaluate_model()
