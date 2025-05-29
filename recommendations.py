"""
Business Recommendations for FreshShop Grocery Chain
This script generates actionable business recommendations based on the analysis:
- Automated weekly restocking quantities
- Optimized promotion scheduling
- Stockout risk alerts
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import sys
import subprocess
from datetime import datetime, timedelta

def load_forecast_data():
    filepath = 'data/sales_forecast.csv'
    
    # Check if forecast data exists
    if not os.path.exists(filepath):
        print(f"Forecast data file {filepath} not found.")
        try:
            modeling_script = os.path.join(os.path.dirname(__file__), 'modeling.py')
            subprocess.run([sys.executable, modeling_script], check=True)
            print("Modeling completed successfully.")
        except subprocess.CalledProcessError as e:
            print(f"Error running modeling script: {e}")
            sys.exit(1)
        if not os.path.exists(filepath):
            print(f"Error: {filepath} still not found after running modeling.")
            sys.exit(1)
    
    forecast_df = pd.read_csv(filepath, parse_dates=['Date'])
    print(f"Loaded forecast data with {forecast_df.shape[0]} rows")
    return forecast_df

def load_stockout_risk_data():
    risk_df = pd.read_csv('data/stockout_risk.csv')
    return risk_df

def load_promo_optimization_data():
    promo_df = pd.read_csv('data/promo_optimization.csv')
    print(f"Loaded promotion optimization data with {promo_df.shape[0]} rows")
    return promo_df

def generate_restocking_recommendations(forecast_df):
    print(" weekly restocking recommendations")
    forecast_df['Week'] = forecast_df['Date'].dt.isocalendar().week
    weekly_forecast = forecast_df.groupby(['Store', 'Week'])['ForecastedSales'].sum().reset_index()
    weekly_forecast['RecommendedStock'] = (weekly_forecast['ForecastedSales'] * 1.2).round().astype(int)

    weekly_forecast.to_csv('data/restocking_recommendations.csv', index=False)
    plt.figure(figsize=(12, 6))
    selected_stores = weekly_forecast['Store'].unique()[:5]
    
    for store in selected_stores:
        store_data = weekly_forecast[weekly_forecast['Store'] == store]
        plt.plot(store_data['Week'], store_data['RecommendedStock'], marker='o', label=f'Store {store}')
    
    plt.title('Weekly Restocking Recommendations')
    plt.xlabel('Week')
    plt.ylabel('Recommended Stock')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('data/restocking_recommendations.png')
    plt.close()
    
    print("Restocking recommendations generated.")
    return weekly_forecast

def generate_promo_recommendations(promo_df):
    print(" promotion scheduling recommendations")
    day_map = {
        1: 'Monday',
        2: 'Tuesday',
        3: 'Wednesday',
        4: 'Thursday',
        5: 'Friday',
        6: 'Saturday',
        7: 'Sunday'
    }
    
    promo_df['DayName'] = promo_df['PromoDay'].map(day_map)
    promo_day_counts = promo_df.groupby('PromoDay')['Store'].count().reset_index()
    promo_day_counts.columns = ['PromoDay', 'StoreCount']
    promo_day_counts['DayName'] = promo_day_counts['PromoDay'].map(day_map)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x='DayName', y='StoreCount', data=promo_day_counts)
    plt.title('Number of Stores by Best Promotion Day')
    plt.xlabel('Day of Week')
    plt.ylabel('Number of Stores')
    plt.tight_layout()
    plt.savefig('data/promo_day_distribution.png')
    plt.close()
    
    promo_df.to_csv('data/enhanced_promo_recommendations.csv', index=False)
    
    print("Promotion recommendations generated.")
    return promo_df

def generate_stockout_alerts(risk_df, forecast_df):
    print(" stockout risk alerts")
    high_risk_stores = risk_df[risk_df['SalesRatio'] > 1.2]['Store'].tolist()
    high_risk_forecast = forecast_df[forecast_df['Store'].isin(high_risk_stores)]
    daily_risk = high_risk_forecast.groupby(['Store', 'Date'])['ForecastedSales'].sum().reset_index()
    store_max_sales = daily_risk.groupby('Store')['ForecastedSales'].max().reset_index()
    store_max_sales.columns = ['Store', 'MaxSales']
    peak_days = pd.merge(daily_risk, store_max_sales, on='Store')
    peak_days = peak_days[peak_days['ForecastedSales'] == peak_days['MaxSales']]
    peak_days['FormattedDate'] = peak_days['Date'].dt.strftime('%Y-%m-%d')
    peak_days[['Store', 'FormattedDate', 'ForecastedSales']].to_csv('data/stockout_alerts.csv', index=False)
    plt.figure(figsize=(12, 6))
    selected_stores = high_risk_stores[:5]
    
    for store in selected_stores:
        store_data = daily_risk[daily_risk['Store'] == store]
        plt.plot(store_data['Date'], store_data['ForecastedSales'], marker='o', label=f'Store {store}')
    
    plt.title('Daily Sales Forecast for High-Risk Stores')
    plt.xlabel('Date')
    plt.ylabel('Forecasted Sales')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('data/high_risk_daily_forecast.png')
    plt.close()
    return peak_days

def generate_executive_summary():
    print("summary")
    forecast_df = load_forecast_data()
    risk_df = load_stockout_risk_data()
    promo_df = load_promo_optimization_data()
    restocking_recs = generate_restocking_recommendations(forecast_df)
    promo_recs = generate_promo_recommendations(promo_df)
    stockout_alerts = generate_stockout_alerts(risk_df, forecast_df)
    
    summary = {
        'total_stores': len(forecast_df['Store'].unique()),
        'forecast_period': f"{forecast_df['Date'].min().strftime('%Y-%m-%d')} to {forecast_df['Date'].max().strftime('%Y-%m-%d')}",
        'high_risk_stores': len(risk_df[risk_df['SalesRatio'] > 1.2]),
        'top_promo_day': promo_recs.groupby('DayName')['Store'].count().idxmax(),
        'avg_recommended_stock_increase': f"{(restocking_recs['RecommendedStock'].sum() / restocking_recs['ForecastedSales'].sum() - 1) * 100:.1f}%"
    }
    
    # Save summary
    with open('data/executive_summary.txt', 'w') as f:
        f.write("FreshShop Grocery Chain - Sales Forecasting Executive Summary\n")
        f.write("==========================================================\n\n")
        f.write(f"Total stores analyzed: {summary['total_stores']}\n")
        f.write(f"Forecast period: {summary['forecast_period']}\n\n")
        f.write("Key Findings:\n")
        f.write(f"- {summary['high_risk_stores']} stores identified at high risk of stockouts\n")
        f.write(f"- Most effective promotion day across stores: {summary['top_promo_day']}\n")
        f.write(f"- Recommended safety stock buffer: {summary['avg_recommended_stock_increase']}\n\n")
        f.write("Recommendations:\n")
        f.write("1. Implement store-specific weekly restocking plans\n")
        f.write("2. Optimize promotion scheduling based on store-specific patterns\n")
        f.write("3. Prepare additional inventory for high-risk stores during peak periods\n")
    
    print("Executive summary generated.")
    return summary

def run_recommendations():
    os.makedirs('data', exist_ok=True)
    summary = generate_executive_summary()
    return summary

if __name__ == "__main__":
    run_recommendations()