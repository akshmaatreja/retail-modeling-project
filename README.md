# Rossmann Store Sales Forecasting Project

## Project Overview
This project provides a comprehensive sales forecasting solution for Rossmann Store Chain, addressing key business challenges:
- Inventory management (over/under-stocking)
- Promotion effectiveness optimization
- Data-driven demand forecasting

The solution includes three main components:
1. **Smart Inventory Management**: Weekly restocking recommendations
2. **Optimized Promotion Strategy**: Store-specific promotion calendars
3. **Stockout Prevention System**: Early warning alerts for high-risk stores

## Dataset
This project uses the [Rossmann Store Sales dataset](https://www.kaggle.com/c/rossmann-store-sales/data) from Kaggle, which contains historical sales data for 1,115 Rossmann drug stores.

## Requirements

### Prerequisites
- Python 3.8+
- Kaggle API credentials
- Git (optional, for cloning the repository)

### Python Dependencies
All required Python packages are listed in `requirements.txt`. Install them using:
```
pip install -r requirements.txt
```

### Kaggle API Setup
1. Create a Kaggle account if you don't have one
2. Go to your Kaggle account settings (https://www.kaggle.com/account)
3. Click on "Create New API Token" to download `kaggle.json`
4. Place this file in `~/.kaggle/` directory:
   ```
   mkdir -p ~/.kaggle
   cp /path/to/downloaded/kaggle.json ~/.kaggle/
   chmod 600 ~/.kaggle/kaggle.json
   ```

## Running the Project

### Option 1: Complete Pipeline
To run the entire pipeline (ETL, EDA, and Modeling):
```
python main.py
```

### Option 2: Individual Components
To run specific components separately:

1. ETL (Extract, Transform, Load):
```
python etl.py
```

2. EDA (Exploratory Data Analysis):
```
python eda.py
```

3. Modeling (XGBoost forecasting):
```
python modeling.py
```

### Viewing Results
- **Data files**: Check the `data/` directory for CSV outputs
- **Visualizations**: View PNG files in the `data/` directory
- **Model metrics**: See `data/model_metrics.txt`
- **Business presentation**: Open `presentation.html` in a web browser

## Model Performance
The XGBoost forecasting model achieves 91.13% accuracy on test data with RMSE of 628.99 units.
