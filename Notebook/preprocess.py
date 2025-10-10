import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.impute import SimpleImputer

# -----------------------------
# Feature groups
# -----------------------------
numeric_features = ['Age', 'Transaction_Amount', 'Account_Balance']
high_card_features = ['Customer_ID', 'Transaction_ID', 'Merchant_ID']
categorical_features = [
    'Gender', 'State', 'City',
    'Bank_Branch', 'Account_Type', 'Transaction_Type',
    'Merchant_Category', 'Transaction_Device',
    'Transaction_Location', 'Device_Type', 'Transaction_Currency'
]
datetime_features = ['Transaction_Date', 'Transaction_Time']

# -----------------------------
# Functions for custom transformations
# -----------------------------
def extract_datetime_features(df):
    df_ = df.copy()
    if 'Transaction_Date' in df_.columns:
        df_['Transaction_Date'] = pd.to_datetime(df_['Transaction_Date'], errors='coerce')
        df_['Transaction_Year'] = df_['Transaction_Date'].dt.year
        df_['Transaction_Month'] = df_['Transaction_Date'].dt.month
        df_['Transaction_Day'] = df_['Transaction_Date'].dt.day
        df_['Transaction_Weekday'] = df_['Transaction_Date'].dt.weekday
    if 'Transaction_Time' in df_.columns:
        df_['Transaction_Time'] = pd.to_datetime(df_['Transaction_Time'], format='%H:%M:%S', errors='coerce')
        df_['Transaction_Hour'] = df_['Transaction_Time'].dt.hour
        df_['Transaction_Minute'] = df_['Transaction_Time'].dt.minute
        df_['Is_Night'] = df_['Transaction_Hour'].apply(lambda x: 1 if x>=22 or x<6 else 0)
    return df_[['Transaction_Year','Transaction_Month','Transaction_Day',
                'Transaction_Weekday','Transaction_Hour','Transaction_Minute','Is_Night']]

def frequency_encode(df, columns):
    df_ = df.copy()
    for col in columns:
        freq_map = df_[col].value_counts(normalize=True)
        df_[col] = df_[col].map(freq_map).fillna(0)
    return df_[columns]

# Wrappers for sklearn
datetime_transformer_func = FunctionTransformer(extract_datetime_features)
high_card_transformer_func = FunctionTransformer(frequency_encode, kw_args={'columns': high_card_features})

# -----------------------------
# Preprocessing pipelines
# -----------------------------
numeric_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

high_card_transformer = Pipeline([
    ('freq_enc', high_card_transformer_func)
])

categorical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

datetime_transformer = Pipeline([
    ('extractor', datetime_transformer_func),
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

preprocessor = ColumnTransformer([
    ('num', numeric_transformer, numeric_features),
    ('high_card', high_card_transformer, high_card_features),
    ('cat', categorical_transformer, categorical_features),
    ('dt', datetime_transformer, datetime_features)
])

# -----------------------------
# Full preprocessing function
# -----------------------------
def fit_preprocess(df):
    full_pipeline = Pipeline([
        ('preprocessor', preprocessor)
    ])
    X_processed = full_pipeline.fit_transform(df)

    # Ensure 2D
    if X_processed.ndim == 1:
        X_processed = X_processed.reshape(-1, 1)

    # Engineered ratio feature
    if 'Transaction_Amount' in df.columns and 'Account_Balance' in df.columns:
        ratio = (df['Transaction_Amount'] / (df['Account_Balance'] + 1e-5)).values.reshape(-1,1)
        X_processed = np.hstack([X_processed, ratio])

    return X_processed, full_pipeline

# -----------------------------
# Transform new data
# -----------------------------
def transform(df, pipeline):
    X_processed = pipeline.transform(df)

    # Ensure 2D
    if X_processed.ndim == 1:
        X_processed = X_processed.reshape(-1, 1)

    if 'Transaction_Amount' in df.columns and 'Account_Balance' in df.columns:
        ratio = (df['Transaction_Amount'] / (df['Account_Balance'] + 1e-5)).values.reshape(-1,1)
        X_processed = np.hstack([X_processed, ratio])

    return X_processed
