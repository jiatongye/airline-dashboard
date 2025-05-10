import pandas as pd
import numpy as np

def clean_data(file_path="Airline Quality Ratings.csv"):
    df = pd.read_csv(file_path)
    
    # Report initial missing values
    print("Initial missing values:\n", df.isnull().sum())
    
    # Fill numerical with median
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())
    
    # Fill categorical with mode and clean strings
    cat_cols = df.select_dtypes(include=['object']).columns
    for col in cat_cols:
        df[col] = df[col].fillna(df[col].mode()[0])
        df[col] = df[col].astype(str).str.strip()
    
    # Clean column names
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_").str.replace("-", "_")
    
    # Drop duplicates
    df = df.drop_duplicates()
    if 'id' in df.columns and df['id'].duplicated().any():
        df = df.drop_duplicates(subset='id')
    
    # Remove invalid ages
    df = df[df['age'] > 0]
    
    # Clean and standardize categorical columns
    cat_cols = ['gender', 'customer_type', 'type_of_travel', 'class', 'satisfaction']
    for col in cat_cols:
        df[col] = df[col].astype(str).str.strip().str.title()
    
    # Standardize satisfaction values
    df['satisfaction'] = df['satisfaction'].replace({
        'Neutral Or Dissatisfied': 'Neutral/Dissatisfied',
        'neutral or dissatisfied': 'Neutral/Dissatisfied',
        'satisfied': 'Satisfied',
        'Satisfied ': 'Satisfied'
    })
    
    # Create binary target
    df['satisfaction_binary'] = df['satisfaction'].apply(lambda x: 1 if x == 'Satisfied' else 0)
    
    # One-hot encode categorical variables
    categorical_cols_to_encode = ['gender', 'customer_type', 'type_of_travel', 'class']
    df_encoded = pd.get_dummies(df, columns=categorical_cols_to_encode, drop_first=True)
    
    # Convert bool dummies to int
    for col in df_encoded.columns:
        if df_encoded[col].dtype == 'bool':
            df_encoded[col] = df_encoded[col].astype(int)
    
    # Remove highly correlated features
    df_numeric = df_encoded.select_dtypes(include=[np.number])
    corr_matrix = df_numeric.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > 0.7)]
    df_reduced = df_numeric.drop(columns=to_drop)
    
    print(f"Dropped {len(to_drop)} highly correlated columns.")
    print("Final shape:", df_reduced.shape)
    print("Missing values after cleaning:\n", df_reduced.isnull().sum())
    
    return df, df_reduced