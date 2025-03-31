import os
import pytest
import joblib
import numpy as np
import pandas as pd
import boto3
import tarfile
from S3 import download_and_extract_model_artifacts  # Ensure S3.py is in your PYTHONPATH


def load_artifacts():
    """
    Downloads and extracts model artifacts from S3 and loads them.
    Returns:
        model, scaler, brand_means, onehot: Loaded artifacts.
    """
    model_dir = download_and_extract_model_artifacts()

    brand_means_path = os.path.join(model_dir, "brand_means.pkl")
    model_path = os.path.join(model_dir, "random_forest_model.pkl")
    scaler_path = os.path.join(model_dir, "scaler.pkl")
    onehot_path = os.path.join(model_dir, "onehotencode.pkl")

    brand_means = joblib.load(brand_means_path)
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    onehot = joblib.load(onehot_path)

    return model, scaler, brand_means, onehot


def preprocess_input(df, brand_means, scaler, onehot):
    """
    Preprocesses the input DataFrame in the same manner as during training/inference.
    Args:
        df: Input DataFrame with raw values.
        brand_means: The target-mean encoding mapping for 'brand'.
        scaler: Fitted MinMaxScaler.
        onehot: Fitted OneHotEncoder.
    Returns:
        X: Preprocessed feature DataFrame ready for model prediction.
    """
    # Map brand to encoded value and drop the original column.
    df["brand_encoded"] = df["brand"].map(brand_means).fillna(brand_means.mean())
    df_processed = df.drop(columns=["brand"])

    # One-hot encode categorical features.
    cat_cols = ["fuel", "seller_type", "transmission"]
    encoded_cat = onehot.transform(df_processed[cat_cols])
    # Use the encoder's fitted feature names. Check for the new method if available.
    if hasattr(onehot, "get_feature_names_out"):
        feature_names = onehot.get_feature_names_out()
    else:
        feature_names = onehot.get_feature_names(cat_cols)
    encoded_cat_df = pd.DataFrame(encoded_cat, columns=feature_names, index=df_processed.index)

    # Scale numerical features.
    num_cols = ['year', 'km_driven', 'owner', 'mileage', 'engine', 'max_power', 'seats']
    scaled_num = scaler.transform(df_processed[num_cols])
    scaled_num_df = pd.DataFrame(scaled_num, columns=num_cols, index=df_processed.index)

    # Combine scaled numerical features, one-hot encoded categorical features, and brand_encoded.
    X = pd.concat([scaled_num_df, encoded_cat_df, df_processed[["brand_encoded"]]], axis=1)
    return X


def test_model_inference():
    # Load the model artifacts
    model, scaler, brand_means, onehot = load_artifacts()

    # Create a sample input DataFrame.
    data = {
        "brand": ["Maruti"],
        "year": [2016],
        "km_driven": [30000],
        "fuel": ["Petrol"],
        "seller_type": ["Individual"],
        "transmission": ["Manual"],
        "owner": [1],
        "mileage": [22.69],
        "engine": [1400.0],
        "max_power": [160.0],
        "seats": [4.0]
    }
    df_input = pd.DataFrame(data)

    # Preprocess the input
    X = preprocess_input(df_input, brand_means, scaler, onehot)
    print("Preprocessed features:")
    print(X.head())

    # Run prediction using the loaded model.
    preds = model.predict(X)
    # If your model was trained using a log transform, reverse it:
    final_preds = np.expm1(preds)

    # Assertions: Check the prediction output is a NumPy array of shape (1,) and is positive.
    assert isinstance(final_preds, np.ndarray), "Prediction output should be a NumPy array"
    assert final_preds.shape == (1,), "Expected a single prediction output"
    assert final_preds[0] > 0, "Predicted price should be positive"

    print("Prediction successful. Predicted value:", final_preds[0])


if __name__ == "__main__":
    test_model_inference()