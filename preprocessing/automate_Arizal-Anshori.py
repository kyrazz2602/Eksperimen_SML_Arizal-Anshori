import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
import os

def load_data(filepath):
    df = pd.read_csv(filepath)
    return df

def handle_missing_values(df):
    for col in df.select_dtypes(include='number').columns:
        df[col].fillna(df[col].median(), inplace=True)
    for col in df.select_dtypes(include='object').columns:
        df[col].fillna(df[col].mode()[0], inplace=True)
    return df

def encode_categorical(df):
    le = LabelEncoder()
    for col in df.select_dtypes(include='object').columns:
        df[col] = le.fit_transform(df[col])
    return df

def scale_features(df, target_col='target'):
    scaler = StandardScaler()
    feature_cols = [c for c in df.columns if c != target_col]
    df[feature_cols] = scaler.fit_transform(df[feature_cols])
    return df

def save_data(df, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Dataset tersimpan di: {output_path}")

def main():
    input_path  = "heart_disease_raw.csv"
    output_path = "preprocessing/heart_disease_preprocessing.csv"

    df = load_data(input_path)
    df = handle_missing_values(df)
    df = encode_categorical(df)
    df = scale_features(df, target_col='Heart Disease Status')
    save_data(df, output_path)

if __name__ == "__main__":
    main()