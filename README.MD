# 🏦 Advancing Fraud Detection through CTGAN-Based Synthetic Transactions

Generate privacy-preserving synthetic bank transaction data using Finance TVAE to improve fraud detection research and model benchmarking.     

---

## 📁 Folder Structure
```
Synthetic-Fraud-AI-Project/
│
├── README.md
├── Final_Report.pdf
├── Fraud_Detection_Slides.pptx
│
├── Model/
│   ├── preprocess_pipeline.pkl          
│   └── best_model_augmented.pkl  
│
├── Data/
│   ├── Bank_Transaction.csv          
│   ├── Augmented_data.csv       
│   └── Synthetic_Bank_Data.csv     
│
└── Notebooks/
    ├── Exploratory Data Analysis (EDA).ipynb                
    ├── Synthetic_Data_Analysis.ipynb     
    ├── Model_Training.ipynb    
    ├── Visualization.ipynb   
    ├── app.py   
    ├── preprocess.py   
    └── catboost_info/
          ├── catboost_training.json
          ├── learn_error.tsv
          └── time_left.tsv
```


## 🎯 Objective

The objective of this project is to generate synthetic financial transaction data using CTGAN, augment it with real-world transactions, and evaluate its impact on fraud detection. By combining synthetic and authentic data, the goal is to address class imbalance and data scarcity while maintaining privacy. Ultimately, this approach aims to improve the performance and accuracy of machine learning models in detecting fraudulent activities.

---

## 👉 Problem Definition & Dataset Selection

**Dataset:** Bank_Transaction.csv

**Features:** 

Categorical Columns:
  - Customer_ID  - Customer_Name  - Gender  - State  - City  - Bank_Branch  - Account_Type  - Transaction_ID  - Transaction_Date  - Transaction_Time  - Merchant_ID  - Transaction_Type  - Merchant_Category  - Transaction_Device  - Transaction_Location  - Device_Type  - Transaction_Currency  - Customer_Contact  - Transaction_Description  - Customer_Email
  
Numerical Columns:
   - Age   - Transaction_Amount   - Account_Balance   - Is_Fraud

**Target:** ( 0 = normal transaction, 1 = fraud )

Class Distribution: (0 = 189912, 1 = 10088)

Percentage Distribution: (0 = 94.956, 1 = 5.044)


### 📌 Problem Statement:

The dataset suffers from a highly imbalanced class distribution, with only 0.17% of transactions labeled as fraud. This imbalance makes it challenging for machine learning models to effectively learn fraudulent patterns, leading to poor recall and a high risk of undetected fraud.

### ⚙️ Library Imports

```
# Standard
import datetime
import time
import collections
from collections import Counter
import warnings
warnings.filterwarnings("ignore")

# Data
import numpy as np
import pandas as pd

# Visualization
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

# Preprocessing & Decomposition
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.manifold import TSNE

# Model Selection & Pipeline
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from imblearn.pipeline import make_pipeline as imbalanced_make_pipeline

# Sampling
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import NearMiss

# Models
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# Metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report
from imblearn.metrics import classification_report_imbalanced

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# TensorFlow
import tensorflow as tf
```

