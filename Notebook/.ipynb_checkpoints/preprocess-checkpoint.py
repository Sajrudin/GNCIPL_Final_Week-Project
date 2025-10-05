import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.base import BaseEstimator, TransformerMixin

# Load dataset
df = pd.read_csv(r'C:\Users\ACER\Desktop\Projects\Finance\Data\Bank_Transaction.csv')

# -----------------------------
# Step 1: Define column groups
# -----------------------------
numeric_features = [
    'Age', 'Transaction_Amount', 'Account_Balance'
]

categorical_features = [
    'Customer_ID', 'Customer_Name', 'Gender', 'State', 'City',
    'Bank_Branch', 'Account_Type', 'Transaction_ID', 'Merchant_ID',
    'Transaction_Type', 'Merchant_Category', 'Transaction_Device',
    'Transaction_Location', 'Device_Type', 'Transaction_Currency',
    'Customer_Contact', 'Customer_Email'
]

datetime_features = [
    'Transaction_Date', 'Transaction_Time'
]

target = 'Is_Fraud'  # For training

# -----------------------------
# Step 2: Define preprocessing pipelines
# -----------------------------

# Numeric pipeline
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),   # Fill missing numeric values
    ('scaler', StandardScaler())                     # Standardize numeric features
])

# Categorical pipeline
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),  # Fill missing with 'missing'
    ('onehot', OneHotEncoder(handle_unknown='ignore'))                      # One-hot encode categorical
])

# Datetime pipeline
def extract_datetime_features(df):
    """
    Custom transformer to extract features from datetime columns.
    Returns year, month, day, hour, minute as separate features.
    """
    df_ = df.copy()
    if 'Transaction_Date' in df_.columns:
        df_['Transaction_Date'] = pd.to_datetime(df_['Transaction_Date'], errors='coerce')
        df_['Transaction_Year'] = df_['Transaction_Date'].dt.year
        df_['Transaction_Month'] = df_['Transaction_Date'].dt.month
        df_['Transaction_Day'] = df_['Transaction_Date'].dt.day
    if 'Transaction_Time' in df_.columns:
        df_['Transaction_Time'] = pd.to_datetime(df_['Transaction_Time'], format='%H:%M:%S', errors='coerce')
        df_['Transaction_Hour'] = df_['Transaction_Time'].dt.hour
        df_['Transaction_Minute'] = df_['Transaction_Time'].dt.minute
    return df_[['Transaction_Year','Transaction_Month','Transaction_Day','Transaction_Hour','Transaction_Minute']]



class DateTimeExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, datetime_cols):
        self.datetime_cols = datetime_cols

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return extract_datetime_features(X)

datetime_transformer = Pipeline(steps=[
    ('extractor', DateTimeExtractor(datetime_features)),
    ('scaler', StandardScaler())
])

# -----------------------------
# Step 3: Combine all transformers
# -----------------------------
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features),
        ('dt', datetime_transformer, datetime_features)
    ]
)

# -----------------------------
# Step 4: Full pipeline with optional PCA
# -----------------------------
full_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('pca', PCA(n_components=0.95, svd_solver='full'))  # Keeps 95% variance
])


print('Preprocess completed!')
# -----------------------------
# Step 5: Fit pipeline (example)
# -----------------------------
# X = df.drop(columns=[target])
# y = df[target]

# full_pipeline.fit(X)
# X_transformed = full_pipeline.transform(X)

# Now `full_pipeline` can also be used for new input from UI:
# new_input_transformed = full_pipeline.transform(new_input_df)
