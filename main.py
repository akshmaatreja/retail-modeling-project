"""
Main script to run the complete pipeline:
1. ETL: Extract, transform, and load data
2. EDA: Exploratory data analysis
3. Modeling: Build and evaluate forecasting models
4. Recommendations: Generate business recommendations
"""

import os
import sys
import time
from datetime import datetime

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from etl import run_etl
from eda import run_eda
from modeling import run_modeling
from recommendations import run_recommendations

def create_project_structure():
    os.makedirs('data', exist_ok=True)

def run_pipeline():
    start_time = time.time()
    create_project_structure()
    print("STEP 1: EXTRACT, TRANSFORM, LOAD")
    etl_output = run_etl()
    print("STEP 2: EXPLORATORY DATA ANALYSIS")
    eda_output = run_eda()
    print("STEP 3: MODELING")
    modeling_output = run_modeling()
    print("STEP 4: BUSINESS RECOMMENDATIONS")
    recommendations_output = run_recommendations()
    
    end_time = time.time()
    execution_time = end_time - start_time
    
    print("\nResults and visualizations are available in the 'data' directory.")
    print("summary is available at 'data/executive_summary.txt'.")

if __name__ == "__main__":
    run_pipeline()