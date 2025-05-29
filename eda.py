"""
Exploratory Data Analysis
This script performs comprehensive EDA on the processed sales data:
- Time-series decompositions
- Promo vs. non-promo lift curves by region
- Store performance analysis
- Seasonal patterns identification
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
import os
import sys
import subprocess

# Set style for plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("viridis")

def load_processed_data(filepath='data/processed_train.csv'):
    # Check if processed data exists
    if not os.path.exists(filepath):
        try:
            etl_script = os.path.join(os.path.dirname(__file__), 'etl.py')
            subprocess.run([sys.executable, etl_script], check=True)
            print("ETL completed successfully.")
        except subprocess.CalledProcessError as e:
            print(f"Error running ETL script: {e}")
            sys.exit(1)
        if not os.path.exists(filepath):
            print(f"Error: {filepath} still not found after running ETL.")
            sys.exit(1)
    
    # Load the data
    df = pd.read_csv(filepath, parse_dates=['Date'])
    print(f"Loaded data with {df.shape[0]} rows and {df.shape[1]} columns")
    return df

def plot_sales_distribution(df):
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    sns.histplot(df['Sales'], kde=True)
    plt.title('Sales Distribution')
    plt.xlabel('Sales')
    
    plt.subplot(1, 2, 2)
    sns.histplot(df['LogSales'], kde=True)
    plt.title('Log-transformed Sales Distribution')
    plt.xlabel('Log Sales')
    
    plt.tight_layout()
    plt.savefig('data/sales_distribution.png')
    plt.close()

def plot_sales_by_store_type(df):
    plt.figure(figsize=(12, 6))
    
    if 'StoreType' in df.columns:
        store_type_sales = df.groupby('StoreType')['Sales'].mean().sort_values(ascending=False)
        
        sns.barplot(x=store_type_sales.index, y=store_type_sales.values)
        plt.title('Average Sales by Store Type')
        plt.xlabel('Store Type')
        plt.ylabel('Average Sales')
    else:
        store_type_cols = [col for col in df.columns if col.startswith('StoreType_')]
        if store_type_cols:
            df['StoreTypeCategory'] = 'Unknown'
            for col in store_type_cols:
                type_name = col.split('_')[1]
                df.loc[df[col] == 1, 'StoreTypeCategory'] = type_name
            
            store_type_sales = df.groupby('StoreTypeCategory')['Sales'].mean().sort_values(ascending=False)
            
            sns.barplot(x=store_type_sales.index, y=store_type_sales.values)
            plt.title('Average Sales by Store Type')
            plt.xlabel('Store Type')
            plt.ylabel('Average Sales')
        else:
            print("Warning: No store type information found in the dataset")
            plt.text(0.5, 0.5, "No store type data available", 
                    horizontalalignment='center', verticalalignment='center')
    
    plt.tight_layout()
    plt.savefig('data/sales_by_store_type.png')
    plt.close()

def plot_sales_by_day_of_week(df):
    """
    Plot sales by day of week
    """
    plt.figure(figsize=(12, 6))
    day_sales = df.groupby('DayOfWeek')['Sales'].mean()
    
    sns.barplot(x=day_sales.index, y=day_sales.values)
    plt.title('Average Sales by Day of Week')
    plt.xlabel('Day of Week (1=Monday, 7=Sunday)')
    plt.ylabel('Average Sales')
    
    plt.tight_layout()
    plt.savefig('data/sales_by_day.png')
    plt.close()

def plot_promo_impact(df):
    plt.figure(figsize=(15, 10))
    plt.subplot(2, 2, 1)
    sns.boxplot(x='Promo', y='Sales', data=df)
    plt.title('Sales Distribution: Promo vs. No Promo')
    plt.xlabel('Promotion Active')
    plt.ylabel('Sales')
    if 'StoreType' in df.columns:
        plt.subplot(2, 2, 2)
        sns.boxplot(x='StoreType', y='Sales', hue='Promo', data=df)
        plt.title('Promo Impact by Store Type')
        plt.xlabel('Store Type')
        plt.ylabel('Sales')
        
        promo_lift = df.groupby(['StoreType', 'Promo'])['Sales'].mean().unstack()
        promo_lift['Lift_Percentage'] = (promo_lift[1] - promo_lift[0]) / promo_lift[0] * 100
        
        plt.subplot(2, 2, 3)
        sns.barplot(x=promo_lift.index, y=promo_lift['Lift_Percentage'])
        plt.title('Promotion Lift Percentage by Store Type')
        plt.xlabel('Store Type')
        plt.ylabel('Lift Percentage (%)')
    else:
        store_type_cols = [col for col in df.columns if col.startswith('StoreType_')]
        if store_type_cols:
            df['StoreTypeCategory'] = 'Unknown'
            for col in store_type_cols:
                type_name = col.split('_')[1]
                df.loc[df[col] == 1, 'StoreTypeCategory'] = type_name
            
            plt.subplot(2, 2, 2)
            sns.boxplot(x='StoreTypeCategory', y='Sales', hue='Promo', data=df)
            plt.title('Promo Impact by Store Type')
            plt.xlabel('Store Type')
            plt.ylabel('Sales')
            
            promo_lift = df.groupby(['StoreTypeCategory', 'Promo'])['Sales'].mean().unstack()
            promo_lift['Lift_Percentage'] = (promo_lift[1] - promo_lift[0]) / promo_lift[0] * 100
            
            plt.subplot(2, 2, 3)
            sns.barplot(x=promo_lift.index, y=promo_lift['Lift_Percentage'])
            plt.title('Promotion Lift Percentage by Store Type')
            plt.xlabel('Store Type')
            plt.ylabel('Lift Percentage (%)')
        else:
            plt.subplot(2, 2, 2)
            plt.text(0.5, 0.5, "No store type data available", 
                    horizontalalignment='center', verticalalignment='center')
            plt.subplot(2, 2, 3)
            plt.text(0.5, 0.5, "No store type data available", 
                    horizontalalignment='center', verticalalignment='center')
    plt.subplot(2, 2, 4)
    sns.boxplot(x='DayOfWeek', y='Sales', hue='Promo', data=df)
    plt.title('Promo Impact by Day of Week')
    plt.xlabel('Day of Week')
    plt.ylabel('Sales')
    
    plt.tight_layout()
    plt.savefig('data/promo_impact.png')
    plt.close()

def time_series_decomposition(df):
    # Aggregate sales by date 
    daily_sales = df.groupby('Date')['Sales'].sum().reset_index()
    daily_sales.set_index('Date', inplace=True)
    decomposition = seasonal_decompose(daily_sales, model='multiplicative', period=7)
    
    plt.figure(figsize=(14, 10))
    plt.subplot(4, 1, 1)
    decomposition.observed.plot()
    plt.title('Observed')
    plt.ylabel('Sales')
    
    plt.subplot(4, 1, 2)
    decomposition.trend.plot()
    plt.title('Trend')
    plt.ylabel('Sales')
    
    plt.subplot(4, 1, 3)
    decomposition.seasonal.plot()
    plt.title('Seasonality')
    plt.ylabel('Factor')
    
    plt.subplot(4, 1, 4)
    decomposition.resid.plot()
    plt.title('Residuals')
    plt.ylabel('Sales')
    
    plt.tight_layout()
    plt.savefig('data/time_series_decomposition.png')
    plt.close()
    
    return decomposition

def analyze_store_performance(df):
    store_performance = df.groupby('Store')['Sales'].agg(['mean', 'std', 'count']).reset_index()
    store_performance.columns = ['Store', 'Avg_Sales', 'Std_Sales', 'Count']
    store_performance = store_performance.sort_values('Avg_Sales', ascending=False)

    plt.figure(figsize=(14, 8))
    plt.subplot(1, 2, 1)
    sns.barplot(x='Store', y='Avg_Sales', data=store_performance.head(10))
    plt.title('Top 10 Performing Stores')
    plt.xlabel('Store ID')
    plt.ylabel('Average Sales')
    plt.xticks(rotation=45)
    
    plt.subplot(1, 2, 2)
    sns.barplot(x='Store', y='Avg_Sales', data=store_performance.tail(10))
    plt.title('Bottom 10 Performing Stores')
    plt.xlabel('Store ID')
    plt.ylabel('Average Sales')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('data/store_performance.png')
    plt.close()
    
    return store_performance

def analyze_seasonal_patterns(df):
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 2, 1)
    monthly_sales = df.groupby('Month')['Sales'].mean()
    sns.barplot(x=monthly_sales.index, y=monthly_sales.values)
    plt.title('Average Sales by Month')
    plt.xlabel('Month')
    plt.ylabel('Average Sales')
    
    plt.subplot(2, 2, 2)
    quarterly_sales = df.groupby('Quarter')['Sales'].mean()
    sns.barplot(x=quarterly_sales.index, y=quarterly_sales.values)
    plt.title('Average Sales by Quarter')
    plt.xlabel('Quarter')
    plt.ylabel('Average Sales')
    state_holiday_cols = [col for col in df.columns if col.startswith('StateHoliday_')]
    plt.subplot(2, 2, 3)
    
    if 'StateHoliday' in df.columns:
        holiday_sales = df.groupby(df['StateHoliday'] != '0')['Sales'].mean()
        sns.barplot(x=['No Holiday', 'Holiday'], y=holiday_sales.values)
    elif state_holiday_cols:
        df['IsHoliday'] = df[state_holiday_cols].sum(axis=1) > 0
        holiday_sales = df.groupby('IsHoliday')['Sales'].mean()
        sns.barplot(x=['No Holiday', 'Holiday'], y=holiday_sales.values)
    else:
        plt.text(0.5, 0.5, "No holiday data available", 
                horizontalalignment='center', verticalalignment='center')
    
    plt.title('Average Sales: Holiday vs. Non-Holiday')
    plt.xlabel('')
    plt.ylabel('Average Sales')
    plt.subplot(2, 2, 4)
    school_holiday_sales = df.groupby('SchoolHoliday')['Sales'].mean()
    sns.barplot(x=['No School Holiday', 'School Holiday'], y=school_holiday_sales.values)
    plt.title('Average Sales: School Holiday vs. Non-School Holiday')
    plt.xlabel('')
    plt.ylabel('Average Sales')
    
    plt.tight_layout()
    plt.savefig('data/seasonal_patterns.png')
    plt.close()

def run_eda():
    print("EDA Pipeline")
    os.makedirs('data', exist_ok=True)
    df = load_processed_data()

    plot_sales_distribution(df)
    plot_sales_by_store_type(df)
    plot_sales_by_day_of_week(df)
    plot_promo_impact(df)
    decomposition = time_series_decomposition(df)
    store_performance = analyze_store_performance(df)
    analyze_seasonal_patterns(df)
    
    return {
        'decomposition': decomposition,
        'store_performance': store_performance
    }

if __name__ == "__main__":
    run_eda()