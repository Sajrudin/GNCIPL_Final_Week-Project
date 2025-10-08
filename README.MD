# 🏦 Advancing Fraud Detection through CTGAN-Based Synthetic Transactions

This project focuses on enhancing fraud detection systems by generating privacy-preserving synthetic bank transactions using CTGAN (Conditional Tabular GAN) .
The synthetic data augments real-world transactions to address class imbalance and data scarcity, leading to more robust and generalizable fraud detection models.

Generate privacy-preserving synthetic bank transaction data using CTGAN to improve fraud detection research and model benchmarking.     

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
│   ├── best_model_augmented.pkl  
│   └── catboost_best_model.pkl  
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

The main goal of this project is to:

    * Generate synthetic financial transactions using CTGAN and Finance TVAE.

    * Combine synthetic and real data to balance the dataset and reduce bias.

    * Train and evaluate multiple machine learning models to measure how synthetic augmentation impacts fraud detection performance.

    * Ensure data privacy and integrity by replacing sensitive data with realistic synthetic equivalents.

---
# 🧩 **Problem Definition**
## **Dataset**

- File: Data/Bank_Transaction.csv

- Target Variable: Is_Fraud

    * 0 → Legitimate Transaction

    * 1 → Fraudulent Transaction

- Class Distribution:

    * Legitimate: 189,912 (≈94.96%)

    * Fraudulent: 10,088 (≈5.04%)

This severe imbalance leads to poor model recall and limited fraud detection capability.

# **Features Overview**

## Categorical Columns:
    - Customer_ID, Gender, City, Bank_Branch, Account_Type, Transaction_ID,
    Transaction_Type, Merchant_Category, Transaction_Device, Transaction_Location,
    Device_Type, Transaction_Currency, Customer_Email, Transaction_Description, etc.

## Numerical Columns:
Age, Transaction_Amount, Account_Balance


# 🧠 **Approach & Methodology**

1. **Data Preprocessing**

    * Handled missing values, encoding, and scaling.

    * Split into train-test sets using Stratified Sampling to preserve class ratios.

    * Implemented a custom preprocessing pipeline (preprocess_pipeline.pkl).

2. **Synthetic Data Generation**

    * Used EVAE Tabular to generate realistic synthetic transactions.

    * Merged synthetic and real data to create the Augmented Dataset.

    * Evaluated synthetic quality metrics (score ≈ 8.5) for realism and privacy.

3. **Model Training and Tuning**

    * Models were trained on both original and augmented datasets.

    * Hyperparameter tuning performed via RandomizedSearchCV .

    * Evaluation metrics: Accuracy, Precision, Recall, and F1-score.

----------------
# 🤖 **Top 3 Performing Models (on Augmented Data)**
| Rank | Model                   | F1 Score   | Recall     | Precision | 
| ---- | ----------------------- | ---------- | ---------- | --------- | 
| 🥇 1 | **CatBoost Classifier** | **0.2868** | **0.8807** | 0.1713    | 
| 🥈 2 | **XGBoost Classifier**  | **0.2861** | 0.8302     | 0.1729    | 
| 🥉 3 | **LightGBM Classifier** | **0.2667** | 0.5772     | 0.1734    | 

---------------
# 🔍 **Key Findings**

    * CatBoost and XGBoost achieved highest recall (>0.83) — ideal for minimizing missed frauds.

    * LightGBM offered a better balance between recall and precision, making it suitable for production scenarios.

    * Linear models (Logistic Regression, SGD) performed decently but struggled with extreme imbalance.

    * Balanced Random Forest, despite high accuracy, failed to detect minority class effectively.

---------------
# 💻 **Technologies Used**
| Category                  | Tools / Libraries                                                                        |
| ------------------------- | ---------------------------------------------------------------------------------------- |
| Data Processing           | `pandas`, `numpy`, `scikit-learn`, `joblib`                                              |
| Synthetic Data Generation | `Tabluar CTGAN`,  `gretel-synthetics`                                                    |
| Machine Learning          | `XGBoost`, `LightGBM`, `CatBoost`, `DecisionTree`, `LogisticRegression`, `SGDClassifier` |
| Visualization             | `matplotlib`, `plotly`, `seaborn`                                                        |
| App Interface             | `Streamlit`                                                                              |
| Environment               | `Python 3.11`, `Anaconda`, `Jupyter Notebooks`, `VS Code`                                |

---------------

# 🚀 **Results & Conclusion**

    * Synthetic data generation using CTGAN improved fraud recall by over 40% compared to models trained on raw data alone.

    * CatBoost emerged as the most recall-efficient model for fraud detection.

    * Combining real and synthetic datasets improved robustness, diversity, and fairness in model training.

    * This approach can serve as a framework for banks and financial institutions seeking privacy-preserving data for fraud detection research and benchmarking.

------------------------
# 👨‍💻 **Author**

**Sajrudin Aalam**  
📍 B.Tech CSE (AI/ML), Graphic Era Hill University (2026)  
🔗[LinkedIn Profile](https://www.linkedin.com/in/sajrudin-aalam-21b861287/)  
📧 aalamsajrudin@gmail.com  


**Rekha Kumari Bheel**  
📍 B.Tech IT, Govt. Women Engineering College,Ajmer

🔗[LinkedIn Profile](https://www.linkedin.com/in/rekha-gunarat-7b9459279/)  
📧 rekha.kumari1928@gmail.com

**Riya Rawat**  
📍 B.Tech CSE, Graphic Era Hill University (2026)  
🔗[LinkedIn Profile](https://www.linkedin.com/in/riya-rawat-bb2b78265/)  

**Payal Dhokane **  
📍 B.E CS, International Centre of Excellence in Engineering & Management    
🔗[LinkedIn Profile](https://www.linkedin.com/in/payal-dhokane-79967b294/)    
📧 payaldhokane282@gmail.com  















