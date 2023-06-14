import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

def main():
    year, month, *_ = map(int, sys.argv[1:])


    # cell copied from homework/ folder
    with open(next(Path().glob("**/model.bin")), 'rb') as f_in:
        dv, model = pickle.load(f_in)


    categorical = ['PULocationID', 'DOLocationID']

    def read_data(filename):
        df = pd.read_parquet(filename)
        df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
        df['duration'] = df.duration.dt.total_seconds() / 60
        df = df[(df.duration >= 1) & (df.duration <= 60)]#.copy()
        df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
        return df


    df = read_data(f'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year:04d}-{month:02d}.parquet')


    dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(dicts)

    y_pred = model.predict(X_val)




    np.std(y_pred)




    df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')

    df_result = (
        pd.DataFrame({
            "pred": y_pred,
            "ride_id": df["ride_id"].to_numpy()})
    )

    #output_file = "df_result.parquet"
    #df_result.to_parquet(
    #    output_file,
    #    engine='pyarrow',
    #    compression=None,
    #    index=False
    #)

    print(np.mean(y_pred))
    return None


if __name__ == "__main__":
    main()
