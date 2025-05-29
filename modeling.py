"""
Sales Forecasting Model for Rossmann Store Chain
This script builds and evaluates an optimized XGBoost regressor to forecast sales:
- Enhanced feature engineering
- Optimized model parameters
- Store-level forecasting
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import joblib
import os
import sys
import subprocess
from datetime import datetime, timedelta

def load_processed_data(filepath='data/processed_train.csv'):
    if not os.path.exists(filepath):
        print(f"Processed data file {filepath} not found.")
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
    
    df = pd.read_csv(filepath, parse_dates=['Date'])
    print(f"Loaded data with {df.shape[0]} rows and {df.shape[1]} columns")
    return df

def engineer_features(df):
    print("Engineering advanced features")
    
    df_enhanced = df.copy()
    df_enhanced['DayOfWeek_sin'] = np.sin(2 * np.pi * df_enhanced['DayOfWeek'] / 7)
    df_enhanced['DayOfWeek_cos'] = np.cos(2 * np.pi * df_enhanced['DayOfWeek'] / 7)
    df_enhanced['Month_sin'] = np.sin(2 * np.pi * df_enhanced['Month'] / 12)
    df_enhanced['Month_cos'] = np.cos(2 * np.pi * df_enhanced['Month'] / 12)
    df_enhanced['Day_sin'] = np.sin(2 * np.pi * df_enhanced['Day'] / 31)
    df_enhanced['Day_cos'] = np.cos(2 * np.pi * df_enhanced['Day'] / 31)
    
    df_enhanced['Promo_Weekend'] = df_enhanced['Promo'] * ((df_enhanced['DayOfWeek'] == 6) | (df_enhanced['DayOfWeek'] == 7))
    df_enhanced['Promo_SchoolHoliday'] = df_enhanced['Promo'] * df_enhanced['SchoolHoliday']
    
    # Create store-specific features
    store_avg = df_enhanced.groupby('Store')['Sales'].mean().reset_index()
    store_avg.columns = ['Store', 'Store_Avg_Sales']
    df_enhanced = pd.merge(df_enhanced, store_avg, on='Store', how='left')
    
    # Create day-of-week specific features
    dow_avg = df_enhanced.groupby(['Store', 'DayOfWeek'])['Sales'].mean().reset_index()
    dow_avg.columns = ['Store', 'DayOfWeek', 'DOW_Avg_Sales']
    df_enhanced = pd.merge(df_enhanced, dow_avg, on=['Store', 'DayOfWeek'], how='left')
    
    # Create month-specific features
    month_avg = df_enhanced.groupby(['Store', 'Month'])['Sales'].mean().reset_index()
    month_avg.columns = ['Store', 'Month', 'Month_Avg_Sales']
    df_enhanced = pd.merge(df_enhanced, month_avg, on=['Store', 'Month'], how='left')
    
    # Create promo-specific features
    promo_avg = df_enhanced.groupby(['Store', 'Promo'])['Sales'].mean().reset_index()
    promo_avg.columns = ['Store', 'Promo', 'Promo_Avg_Sales']
    df_enhanced = pd.merge(df_enhanced, promo_avg, on=['Store', 'Promo'], how='left')
    
    df_enhanced['CompetitionDistance_Log'] = np.log1p(df_enhanced['CompetitionDistance'])
    df_enhanced['CompetitionOpen_Squared'] = df_enhanced['CompetitionOpen'] ** 2
    
    if 'Customers' in df_enhanced.columns:
        df_enhanced['SalesPerCustomer'] = df_enhanced['Sales'] / df_enhanced['Customers'].replace(0, 1)
        df_enhanced['SalesPerCustomer'] = df_enhanced['SalesPerCustomer'].fillna(0)

    df_enhanced['Sales_Lag1'] = np.nan
    df_enhanced['Sales_Lag7'] = np.nan
    df_enhanced['Sales_Lag30'] = np.nan
    df_enhanced['Sales_Roll7'] = np.nan
    df_enhanced['Sales_Roll30'] = np.nan
    
    for store in df_enhanced['Store'].unique():
        store_mask = df_enhanced['Store'] == store
        store_data = df_enhanced.loc[store_mask].sort_values('Date')

        df_enhanced.loc[store_mask, 'Sales_Lag1'] = store_data['Sales'].shift(1)
        df_enhanced.loc[store_mask, 'Sales_Lag7'] = store_data['Sales'].shift(7)
        df_enhanced.loc[store_mask, 'Sales_Lag30'] = store_data['Sales'].shift(30)
        
        df_enhanced.loc[store_mask, 'Sales_Roll7'] = store_data['Sales'].rolling(window=7, min_periods=1).mean()
        df_enhanced.loc[store_mask, 'Sales_Roll30'] = store_data['Sales'].rolling(window=30, min_periods=1).mean()
    
    lag_cols = ['Sales_Lag1', 'Sales_Lag7', 'Sales_Lag30', 'Sales_Roll7', 'Sales_Roll30']
    for col in lag_cols:
        df_enhanced[col] = df_enhanced[col].fillna(df_enhanced.groupby('Store')['Sales'].transform('mean'))
    
    print(f"Created {df_enhanced.shape[1] - df.shape[1]} new features")
    return df_enhanced

def prepare_features(df):
    
    df_enhanced = engineer_features(df)
    base_features = [
        'Store', 'DayOfWeek', 'Open', 'Promo', 'SchoolHoliday',
        'Year', 'Month', 'Day', 'WeekOfYear', 'IsMonthStart', 'IsMonthEnd',
        'CompetitionDistance', 'CompetitionOpen'
    ]
    store_type_cols = [col for col in df_enhanced.columns if col.startswith('StoreType_')]
    assortment_cols = [col for col in df_enhanced.columns if col.startswith('Assortment_')]
    state_holiday_cols = [col for col in df_enhanced.columns if col.startswith('StateHoliday_')]
    
    advanced_features = [
        'Sales_Lag1', 'Sales_Lag7', 'Sales_Lag30', 
        'Sales_Roll7', 'Sales_Roll30',
        'DayOfWeek_sin', 'DayOfWeek_cos', 'Month_sin', 'Month_cos', 'Day_sin', 'Day_cos',
        'Promo_Weekend', 'Promo_SchoolHoliday',
        'Store_Avg_Sales', 'DOW_Avg_Sales', 'Month_Avg_Sales', 'Promo_Avg_Sales',
        'CompetitionDistance_Log', 'CompetitionOpen_Squared'
    ]
    
    features = base_features + store_type_cols + assortment_cols + state_holiday_cols + advanced_features

    if 'SalesPerCustomer' in df_enhanced.columns:
        features.append('SalesPerCustomer')
    
    target = 'Sales'
    df_enhanced = df_enhanced[df_enhanced['Open'] == 1]
    
    # Splitting the data 
    X = df_enhanced[features]
    y = df_enhanced[target]

    train_end_date = df_enhanced['Date'].max() - timedelta(days=60)
    
    X_train = X[df_enhanced['Date'] <= train_end_date]
    y_train = y[df_enhanced['Date'] <= train_end_date]
    
    X_test = X[df_enhanced['Date'] > train_end_date]
    y_test = y[df_enhanced['Date'] > train_end_date]
    
    print(f"Training data: {X_train.shape[0]} samples, {X_train.shape[1]} features")
    print(f"Testing data: {X_test.shape[0]} samples, {X_test.shape[1]} features")
    
    return X_train, X_test, y_train, y_test, features, df_enhanced

def train_xgboost_model(X_train, y_train):
    print("Training XGBoost model")
    
    # Define XGBoost parameters for high accuracy
    params = {
        'objective': 'reg:squarederror',
        'max_depth': 12,
        'learning_rate': 0.03,
        'n_estimators': 1000,
        'subsample': 0.9,
        'colsample_bytree': 0.9,
        'min_child_weight': 3,
        'gamma': 0.1,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'random_state': 42
    }
    
    model = xgb.XGBRegressor(**params)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    print(" XGBoost model mterics")

    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    accuracy = 1 - (rmse / y_test.mean())
    accuracy_pct = accuracy * 100
    
    print(f"Mean Absolute Error: {mae:.2f}")
    print(f"Root Mean Squared Error: {rmse:.2f}")
    print(f"R² Score: {r2:.4f}")
    print(f"Accuracy: {accuracy_pct:.2f}%")
    
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.3)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel('Actual Sales')
    plt.ylabel('Predicted Sales')
    plt.title('XGBoost: Actual vs Predicted Sales')
    plt.tight_layout()
    plt.savefig('data/xgboost_actual_vs_predicted.png')
    plt.close()
    
    return {
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'accuracy': accuracy_pct
    }

def plot_feature_importance(model, features):

    importance = model.feature_importances_
    
    feature_importance = pd.DataFrame({
        'Feature': features,
        'Importance': importance
    })
    
    feature_importance = feature_importance.sort_values('Importance', ascending=False)
    
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Importance', y='Feature', data=feature_importance.head(15))
    plt.title('XGBoost: Feature Importance')
    plt.tight_layout()
    plt.savefig('data/xgboost_feature_importance.png')
    plt.close()
    
    return feature_importance

def forecast_next_month(model, df_enhanced, features):

    print("Generating forecasts for the next month")

    last_date = df_enhanced['Date'].max()
    next_month_start = last_date + timedelta(days=1)
    next_month_end = next_month_start + timedelta(days=30)
    stores = df_enhanced['Store'].unique()
    
    forecast_data = []
    
    # For each store, create forecast entries for each day
    for store in stores:
        store_data = df_enhanced[df_enhanced['Store'] == store].sort_values('Date').tail(30).copy()
        current_date = next_month_start
        day_counter = 1
        
        while current_date <= next_month_end:
            new_record = store_data.iloc[-1].copy()
            new_record['Date'] = current_date
            new_record['Year'] = current_date.year
            new_record['Month'] = current_date.month
            new_record['Day'] = current_date.day
            new_record['DayOfWeek'] = current_date.weekday() + 1
            new_record['WeekOfYear'] = current_date.isocalendar()[1]
            new_record['IsMonthStart'] = 1 if current_date.day == 1 else 0
            new_record['IsMonthEnd'] = 1 if (current_date + timedelta(days=1)).month != current_date.month else 0
            new_record['DayOfWeek_sin'] = np.sin(2 * np.pi * new_record['DayOfWeek'] / 7)
            new_record['DayOfWeek_cos'] = np.cos(2 * np.pi * new_record['DayOfWeek'] / 7)
            new_record['Month_sin'] = np.sin(2 * np.pi * new_record['Month'] / 12)
            new_record['Month_cos'] = np.cos(2 * np.pi * new_record['Month'] / 12)
            new_record['Day_sin'] = np.sin(2 * np.pi * new_record['Day'] / 31)
            new_record['Day_cos'] = np.cos(2 * np.pi * new_record['Day'] / 31)
            
            new_record['Open'] = 1
            new_record['SchoolHoliday'] = 0
            state_holiday_cols = [col for col in df_enhanced.columns if col.startswith('StateHoliday_')]
            for col in state_holiday_cols:
                if col == 'StateHoliday_0':
                    new_record[col] = 1
                else:
                    new_record[col] = 0
            
            new_record['Promo_Weekend'] = new_record['Promo'] * ((new_record['DayOfWeek'] == 6) | (new_record['DayOfWeek'] == 7))
            new_record['Promo_SchoolHoliday'] = new_record['Promo'] * 0  # No school holiday
            
            forecast_data.append(new_record)
            current_date += timedelta(days=1)
            day_counter += 1
    forecast_df = pd.DataFrame(forecast_data)
    
    X_forecast = forecast_df[features]
    forecast_df['ForecastedSales'] = model.predict(X_forecast)
    
    forecast_df['ForecastedSales'] = forecast_df['ForecastedSales'].round().astype(int)
    forecast_df[['Store', 'Date', 'ForecastedSales']].to_csv('data/sales_forecast.csv', index=False)
    
    print("Forecasts generated and saved to 'data/sales_forecast.csv'")
    return forecast_df

def identify_stockout_risk(forecast_df, df):
    print("Identifying stores at risk of stockouts")
    
    avg_sales = df.groupby('Store')['Sales'].mean().reset_index()
    avg_sales.columns = ['Store', 'AvgSales']
    forecast_summary = forecast_df.groupby('Store')['ForecastedSales'].mean().reset_index()
    forecast_summary = pd.merge(forecast_summary, avg_sales, on='Store')
    forecast_summary['SalesRatio'] = forecast_summary['ForecastedSales'] / forecast_summary['AvgSales']
    
    high_risk = forecast_summary[forecast_summary['SalesRatio'] > 1.2].sort_values('SalesRatio', ascending=False)
    high_risk.to_csv('data/stockout_risk.csv', index=False)
    
    plt.figure(figsize=(12, 6))
    sns.barplot(x='Store', y='SalesRatio', data=high_risk.head(10))
    plt.axhline(y=1.0, color='r', linestyle='--')
    plt.title('Top 10 Stores at Risk of Stockouts')
    plt.xlabel('Store ID')
    plt.ylabel('Forecasted Sales / Average Sales')
    plt.tight_layout()
    plt.savefig('data/stockout_risk.png')
    plt.close()
    
    print(f"Identified {len(high_risk)} stores at risk of stockouts.")
    return high_risk

def optimize_promo_scheduling(model, df, features):
    print("Optimizing promotion scheduling")
    stores = df['Store'].unique()
    optimization_results = []
    
    # For each store, simulate promotions on different days
    for store in stores[:10]:  # Limit to 10 stores for demonstration
        store_data = df[(df['Store'] == store) & (df['Date'] >= df['Date'].max() - timedelta(days=7))].copy()
        
        for day in range(1, 8):
            sim_data = store_data.copy()
            sim_data.loc[sim_data['DayOfWeek'] == day, 'Promo'] = 1
            if 'Promo_Weekend' in sim_data.columns:
                sim_data['Promo_Weekend'] = sim_data['Promo'] * ((sim_data['DayOfWeek'] == 6) | (sim_data['DayOfWeek'] == 7))
            
            if 'Promo_SchoolHoliday' in sim_data.columns:
                sim_data['Promo_SchoolHoliday'] = sim_data['Promo'] * sim_data['SchoolHoliday']
            
            X_sim = sim_data[features]
            sim_data['SimulatedSales'] = model.predict(X_sim)
            total_sales = sim_data['SimulatedSales'].sum()
            optimization_results.append({
                'Store': store,
                'PromoDay': day,
                'TotalSales': total_sales
            })
    
    optimization_df = pd.DataFrame(optimization_results)
    best_promo_days = optimization_df.loc[optimization_df.groupby('Store')['TotalSales'].idxmax()]
    
    best_promo_days.to_csv('data/promo_optimization.csv', index=False)
    plt.figure(figsize=(14, 8))
    
    for i, store in enumerate(stores[:5]):
        plt.subplot(2, 3, i+1)
        store_data = optimization_df[optimization_df['Store'] == store]
        sns.barplot(x='PromoDay', y='TotalSales', data=store_data)
        plt.title(f'Store {store}')
        plt.xlabel('Day of Week')
        plt.ylabel('Total Sales')
    
    plt.tight_layout()
    plt.savefig('data/promo_optimization.png')
    plt.close()
    
    print("Promotion optimization completed.")
    return best_promo_days

def run_modeling():
    print("Modeling Pipeline")
    
    os.makedirs('data', exist_ok=True)
    df = load_processed_data()
    X_train, X_test, y_train, y_test, features, df_enhanced = prepare_features(df)
    xgb_model = train_xgboost_model(X_train, y_train)
    
    metrics = evaluate_model(xgb_model, X_test, y_test)
    feature_importance = plot_feature_importance(xgb_model, features)
    
    joblib.dump(xgb_model, 'data/best_model.pkl')
    forecast_df = forecast_next_month(xgb_model, df_enhanced, features)
    high_risk_stores = identify_stockout_risk(forecast_df, df)
    best_promo_days = optimize_promo_scheduling(xgb_model, df_enhanced, features)
    print(f"Final model accuracy: {metrics['accuracy']:.2f}%")

    with open('data/model_metrics.txt', 'w') as f:
        f.write(f"Model: XGBoost\n")
        f.write(f"Accuracy: {metrics['accuracy']:.2f}%\n")
        f.write(f"RMSE: {metrics['rmse']:.2f}\n")
        f.write(f"MAE: {metrics['mae']:.2f}\n")
        f.write(f"R² Score: {metrics['r2']:.4f}\n")
    
    return {
        'metrics': metrics,
        'forecast_df': forecast_df,
        'high_risk_stores': high_risk_stores,
        'best_promo_days': best_promo_days
    }

if __name__ == "__main__":
    run_modeling()