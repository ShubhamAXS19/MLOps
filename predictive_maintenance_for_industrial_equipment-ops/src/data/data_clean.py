CREATE TABLE sensor_data_cleaned AS
SELECT *,
    CASE
        WHEN value IS NULL THEN MEDIAN(value) OVER (PARTITION BY sensor_id)
        ELSE value
    END AS value_imputed
FROM sensor_data_raw
WHERE ABS(value - AVG(value) OVER (PARTITION BY sensor_id)) < 3 * STDDEV(value) OVER (PARTITION BY sensor_id);
