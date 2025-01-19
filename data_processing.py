import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# Load data from uploaded file
def load_data(uploaded_file):
    try:
        if uploaded_file.name.endswith('.csv'):
            return pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith('.xlsx'):
            return pd.read_excel(uploaded_file)
        elif uploaded_file.name.endswith('.json'):
            return pd.read_json(uploaded_file)
        else:
            return None
    except Exception:
        return None

# Handle missing data using imputation
def handle_missing_data(data):
    numerical_cols = data.select_dtypes(include=['float64', 'int64']).columns
    categorical_cols = data.select_dtypes(include=['object']).columns
    
    # Impute missing values in numerical columns using mean
    numerical_imputer = SimpleImputer(strategy="mean")
    data[numerical_cols] = numerical_imputer.fit_transform(data[numerical_cols])
    
    # Impute missing values in categorical columns using the most frequent value
    categorical_imputer = SimpleImputer(strategy="most_frequent")
    data[categorical_cols] = categorical_imputer.fit_transform(data[categorical_cols])
    
    return data

# Normalize numerical columns using StandardScaler
def normalize_numerical_columns(data):
    numerical_cols = data.select_dtypes(include=['float64', 'int64']).columns
    scaler = StandardScaler()
    data[numerical_cols] = scaler.fit_transform(data[numerical_cols])
    return data
