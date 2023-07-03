
import datetime
import time
import random
import logging 
from pathlib import Path

import pyarrow.parquet as pq
import pandas as pd
import numpy as np
import pyarrow
import psycopg2
import joblib

from prefect import task, flow
from prefect.artifacts import create_markdown_artifact
from evidently.metrics import (
    ColumnDriftMetric,
    ColumnQuantileMetric, 
    DatasetDriftMetric, 
    DatasetMissingValuesMetric,
)
from evidently import ColumnMapping
from evidently.report import Report
from sqlalchemy import create_engine, types, Engine
from pandas.tseries.frequencies import to_offset


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s]: %(message)s")

num_features = ['passenger_count', 'trip_distance', 'fare_amount', 'total_amount']
cat_features = ['PULocationID', 'DOLocationID']
column_mapping = ColumnMapping(
    prediction='prediction',
    numerical_features=num_features,
    categorical_features=cat_features,
    target=None
)

report = Report(metrics = [
    ColumnDriftMetric(column_name='prediction'),
    ColumnQuantileMetric(column_name='fare_amount', quantile=0.5),
    DatasetDriftMetric(),
    DatasetMissingValuesMetric(),
])


@task(log_prints=True)
def parquet_to_db(
    filename: str,
    engine: Engine,
):
    parquet_file = pq.ParquetFile(filename)
    engine.connect()
    for i, d in enumerate(parquet_file.iter_batches(batch_size=10000)):
        dd: pd.DataFrame = d.to_pandas()
        if i == 0:
            dd.to_sql(name="rides", con=engine, if_exists='replace')
        else:
            dd.to_sql(name="rides", con=engine, if_exists='append')


@task(log_prints=True)
def metrics_to_db(
    start_date: pd.Period,
    end_date: pd.Period,
    engine: Engine,
):
    engine.connect()
    (
        pd.DataFrame({
            "timestamp": [],
            "prediction_drift": [],
            "fare_amount_quantile05": [],
            "num_drifted_columns": [],
            "share_missing_values": []})
        .to_sql(
            name="metrics",
            con=engine,
            if_exists='replace',
            dtype={
                "timestamp": types.DATE,
                "prediction_drift": types.FLOAT,
                "fare_amount_quantile05": types.FLOAT,
                "num_drifted_columns": types.INTEGER,
                "share_missing_values": types.INTEGER})
    )

    for date in pd.date_range(start=start_date, end=end_date, freq="D"):
        query = f"""
        SELECT *
        FROM rides
        WHERE 
            '{str(date)}'<= lpep_pickup_datetime AND
            lpep_pickup_datetime < '{str(date + to_offset("D"))}'
        """
        current = pd.read_sql(query, con=engine).assign(prediction=0)
        reference_data = current

        report.run(
            reference_data=reference_data,
            current_data=current,
		    column_mapping=column_mapping,
        )

        result = report.as_dict()
        print(result)

        prediction_drift = result['metrics'][0]['result']['drift_score']
        fare_amount_quantile05 = result['metrics'][1]['result']['current']['value']
        num_drifted_columns = result['metrics'][2]['result']['number_of_drifted_columns']
        share_missing_values = result['metrics'][3]['result']['current']['share_of_missing_values']

        (
            pd.DataFrame(dict(zip(
                ("timestamp", "prediction_drift", "fare_amount_quantile05", "num_drifted_columns", "share_missing_values"),
                ([str(date)], [prediction_drift], [fare_amount_quantile05], [num_drifted_columns], [share_missing_values]))))
            .to_sql(
                name="metrics",
                con=engine,
                if_exists='append')
        )


@task(log_prints=True)
def save_max_quantile(
    start_date: pd.Period,
    end_date: pd.Period,
    engine: Engine,
):
    engine.connect()
    query = f"""
    SELECT max(fare_amount_quantile05)
    FROM metrics
    WHERE 
        '{str(start_date)}'<= timestamp AND
        timestamp <= '{str(end_date)}'
    """
    current = pd.read_sql(query, con=engine)
    create_markdown_artifact(
        key=f"max-daily-quantile-{str(start_date)}--{str(end_date)}",
        markdown=current.to_markdown(),
        description="Max daily quantile.",
    )

@flow(log_prints=True, validate_parameters=False)
def prep_db(
    start_date: pd.Period,
    end_date: pd.Period,
    filename: str,
    db: str,
    port: str|int = 5432,
    host: str = "127.0.0.1",
    user: str = "postgres",
    password: str = "postgres",
):
    engine = create_engine(f'postgresql://{user}:{password}@{host}:{port}/{db}')
    parquet_to_db(filename, engine)
    metrics_to_db(start_date, end_date, engine)
    save_max_quantile(start_date, end_date, engine)
