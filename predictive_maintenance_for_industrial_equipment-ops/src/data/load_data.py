# src/data_ingestion.py
import pandas as pd
from sqlalchemy import create_engine

def get_db_connection():
    # Replace with your AWS RDS credentials
    engine = create_engine('postgresql://username:password@rds_endpoint:port/dbname')
    return engine.connect()

def fetch_sensor_data(query: str):
    conn = get_db_connection()
    df = pd.read_sql(query, conn)
    conn.close()
    return df

if __name__ == "__main__":
    query = "SELECT * FROM sensor_data_raw;"
    data = fetch_sensor_data(query)
    data.to_csv('data/raw/sensor_data.csv', index=False)
