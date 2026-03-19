1. Data Preprocessing + Feature Engineering

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

def validate_dataset(df, name=""):
    print(f"\n=== Dataset Validation: {name} ===")
    print("Rows:", df.shape[0], " | Columns:", df.shape[1])
    print("Total Missing:", df.isnull().sum().sum())
    print("Duplicate Rows:", df.duplicated().sum())

    return True


def impute_missing(df):
    df_clean = df.copy()
    num_cols = df_clean.select_dtypes(include=np.number).columns
    for col in num_cols:
        df_clean[col] = df_clean[col].fillna(df_clean[col].median())
    return df_clean


def handle_outliers(df):
    df_new = df.copy()
    num_cols = df_new.select_dtypes(include=np.number).columns

    for col in num_cols:
        q1, q3 = df_new[col].quantile(0.25), df_new[col].quantile(0.75)
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        df_new[col] = df_new[col].clip(lower, upper)

    return df_new


def feature_engineering(df):
    df = df.copy()

    if 'bmi' in df.columns:
        df['bmi_category'] = pd.cut(
            df['bmi'],
            bins=[0, 18.5, 24.9, 29.9, 100],
            labels=['Underweight', 'Normal', 'Overweight', 'Obese']
        )

    if 'daily_steps' in df.columns:
        df['activity_level'] = pd.cut(
            df['daily_steps'],
            bins=[0, 5000, 8000, 12000, 50000],
            labels=['Low', 'Average', 'Active', 'Highly Active']
        )

    if {'bmi', 'age'}.issubset(df.columns):
        df['health_risk_score'] = (
            df['bmi'] * 0.4 + df['age'] * 0.6
        ) / 10

    return df


def scale_numeric(df):
    df_scaled = df.copy()
    numeric_cols = df_scaled.select_dtypes(include=np.number).columns
    scaler = StandardScaler()
    df_scaled[numeric_cols] = scaler.fit_transform(df_scaled[numeric_cols])
    return df_scaled
     
