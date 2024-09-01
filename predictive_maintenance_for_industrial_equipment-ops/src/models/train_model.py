# src/train_model.py
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sqlalchemy import create_engine
import mlflow

def fetch_features(query: str):
    conn = get_db_connection()
    df = pd.read_sql(query, conn)
    conn.close()
    return df

def train_random_forest(X, y):
    rf = RandomForestClassifier()
    rf.fit(X, y)
    return rf

if __name__ == "__main__":
    mlflow.start_run()
    
    query = "SELECT * FROM sensor_features;"
    df = fetch_features(query)

    X = df.drop('target', axis=1)
    y = df['target']

    model = train_random_forest(X, y)
    
    # Log model and metrics in MLflow
    mlflow.sklearn.log_model(model, "model")
    mlflow.log_params({"n_estimators": 100, "criterion": "gini"})
    mlflow.log_metrics({"accuracy": model.score(X, y)})
    
    print(classification_report(y, model.predict(X)))
    
    mlflow.end_run()
