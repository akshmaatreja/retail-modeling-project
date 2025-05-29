"""
ETL Process
This script handles the extraction, transformation, and loading of data from multiple sources:
- Historic sales data
- Store information
- Promotions calendar
- External factors (holidays, weather)
"""

import os
import sys
import zipfile
import subprocess
from datetime import datetime
import pandas as pd
import numpy as np
from kaggle.api.kaggle_api_extended import KaggleApi


def download_rossmann_data(data_dir: str = 'data') -> None:
    os.makedirs(data_dir, exist_ok=True)
    zip_path = os.path.join(data_dir, 'rossmann-store-sales.zip')

    if os.path.exists(zip_path):
        print("Dataset zip already exists. Skipping download.")
    else:
        api = KaggleApi()
        api.authenticate()
        try:
            api.competition_download_files(
                'rossmann-store-sales', path=data_dir, quiet=False
            )
        except Exception as e:
            try:
                subprocess.check_call([
                    'kaggle', 'competitions', 'download',
                    '-c', 'rossmann-store-sales',
                    '-p', data_dir
                ])
            except subprocess.CalledProcessError as cli_e:
                sys.exit(1)

    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(data_dir)


def load_data(data_dir: str = 'data') -> (pd.DataFrame, pd.DataFrame, pd.DataFrame):
    train_path = os.path.join(data_dir, 'train.csv')
    test_path = os.path.join(data_dir, 'test.csv')
    store_path = os.path.join(data_dir, 'store.csv')

    train = pd.read_csv(train_path, parse_dates=['Date'])
    print(f"Loading {test_path}")
    test = pd.read_csv(test_path, parse_dates=['Date'])
    store = pd.read_csv(store_path)

    df = train.merge(store, on='Store', how='left')
    print(f"Merged train+store shape: {df.shape}")
    return df, test, store


def transform_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Day'] = df['Date'].dt.day
    df['WeekOfYear'] = df['Date'].dt.isocalendar().week.astype(int)
    df['DayOfWeek'] = df['Date'].dt.dayofweek + 1
    df['IsMonthStart'] = df['Date'].dt.is_month_start.astype(int)
    df['IsMonthEnd'] = df['Date'].dt.is_month_end.astype(int)
    df['Quarter'] = df['Date'].dt.quarter
    
    df['CompetitionOpenSinceYear'].fillna(df['Year'], inplace=True)
    df['CompetitionOpenSinceMonth'].fillna(df['Month'], inplace=True)
    df['CompetitionOpen'] = (
        12 * (df['Year'] - df['CompetitionOpenSinceYear']) +
        (df['Month'] - df['CompetitionOpenSinceMonth'])
    ).clip(lower=0)
    month_map = {
        'Jan':1,'Feb':2,'Mar':3,'Apr':4,'May':5,'Jun':6,
        'Jul':7,'Aug':8,'Sept':9,'Oct':10,'Nov':11,'Dec':12
    }
    def parse_promo_interval(x):
        if pd.isna(x):
            return []
        return [month_map.get(m.strip(), 0) for m in x.split(',')]

    df['PromoMonths'] = df['PromoInterval'].apply(parse_promo_interval)
    df['IsPromoMonth'] = df.apply(lambda r: int(r['Month'] in r['PromoMonths']), axis=1)
    df['Promo2SinceYear'].fillna(df['Year'], inplace=True)
    df['Promo2SinceWeek'].fillna(df['WeekOfYear'], inplace=True)
    df['Promo2Open'] = (
        12 * (df['Year'] - df['Promo2SinceYear']) +
        (df['WeekOfYear'] - df['Promo2SinceWeek']) / 4.0
    ).clip(lower=0)

    print("Filling missing values and encoding")
    df['CompetitionDistance'].fillna(df['CompetitionDistance'].median(), inplace=True)
    df = pd.get_dummies(df, columns=['StoreType','Assortment','StateHoliday'], drop_first=True)

    print("=Transforming targets and metrics")
    df['LogSales'] = np.log1p(df['Sales'])
    df['LogCustomers'] = np.log1p(df['Customers'])
    df['SalesPerCustomer'] = (df['Sales'] / df['Customers']).replace([np.inf, -np.inf], 0).fillna(0)
    return df


def save_processed_data(df: pd.DataFrame, data_dir: str = 'data', filename: str = 'processed_train.csv') -> str:
    path = os.path.join(data_dir, filename)
    df.to_csv(path, index=False)
    print(f"Saved processed data to {path}")
    return path


def run_etl(data_dir: str = 'data') -> None:
    print("ETL Start")
    download_rossmann_data(data_dir)
    df, test, store = load_data(data_dir)
    proc_df = transform_data(df)
    save_processed_data(proc_df, data_dir, 'processed_train.csv')
    test.to_csv(os.path.join(data_dir, 'processed_test.csv'), index=False)
    store.to_csv(os.path.join(data_dir, 'processed_store.csv'), index=False)
    print("ETL Ended")


if __name__ == '__main__':
    run_etl('data')