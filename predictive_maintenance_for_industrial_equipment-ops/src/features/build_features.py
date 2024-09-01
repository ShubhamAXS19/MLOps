
CREATE TABLE sensor_features AS
SELECT sensor_id, timestamp,
    AVG(value_imputed) OVER (PARTITION BY sensor_id ORDER BY timestamp ROWS BETWEEN 5 PRECEDING AND CURRENT ROW) AS rolling_avg_5,
    MAX(value_imputed) OVER (PARTITION BY sensor_id ORDER BY timestamp ROWS BETWEEN 5 PRECEDING AND CURRENT ROW) AS rolling_max_5
FROM sensor_data_cleaned;
