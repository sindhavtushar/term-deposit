import numpy as np
import pandas as pd
import joblib
import os
import gdown  # Install via: pip install gdown

MODEL_PATH = "ensemble_model.pkl"

GDRIVE_FILE_ID = "1TMJLJGe_aZD58jaM25GXuVBw1tFf0PCo"  # Your Google Drive file ID , https://drive.google.com/file/d/1TMJLJGe_aZD58jaM25GXuVBw1tFf0PCo/view?usp=sharing

def download_model():
    """Download the model from Google Drive if it's not available locally."""
    if not os.path.exists(MODEL_PATH):
        print("📥 Downloading model from Google Drive...")
        gdown.download(f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}", MODEL_PATH, quiet=False)
    else:
        print("✅ Model already exists locally.")

# Ensure model is downloaded before loading
download_model()

# Load the model
if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
    print("✅ Model loaded successfully.")
else:
    raise FileNotFoundError("🚨 Model download failed. Check Google Drive link or internet connection.")

# Define expected numeric and categorical features
NUMERIC_FEATURES = ["age", "balance", "day", "duration", "campaign", "pdays", "previous"]
CATEGORICAL_FEATURES = ["job", "marital", "education", "default", "housing", "loan", "contact", "month", "poutcome"]

# Load the feature names used during model training (if available)
ENCODER_PATH = "encoder.joblib"  # Save encoder if one was used during training

if os.path.exists(ENCODER_PATH):
    encoder = joblib.load(ENCODER_PATH)  # Load trained encoder
    TRAINED_FEATURES = encoder.get_feature_names_out()  # Ensure same feature names are used
else:
    TRAINED_FEATURES = None  # Use only if encoding was done in training

def preprocess_input(user_input):
    """
    Convert user input into the correct format for the model.
    
    :param user_input: Dictionary of form data
    :return: Processed Pandas DataFrame
    """
    try:
        # Convert numeric fields
        numeric_data = {feature: int(user_input.get(feature, 0)) for feature in NUMERIC_FEATURES}

        # Convert categorical fields
        categorical_data = {feature: user_input.get(feature, "unknown") for feature in CATEGORICAL_FEATURES}
        df = pd.DataFrame([{**numeric_data, **categorical_data}])

        # Apply One-Hot Encoding (OHE) if model was trained with it
        if TRAINED_FEATURES is not None:
            df = encoder.transform(df)  # Apply trained encoder
            df = pd.DataFrame(df, columns=TRAINED_FEATURES)  # Convert to DataFrame
        else:
            # Apply categorical conversion for models that require it
            for col in CATEGORICAL_FEATURES:
                df[col] = df[col].astype("category")

        return df
    except ValueError as e:
        print("Error in preprocessing:", e)
        return None  # Handle invalid input gracefully

def predict_subscription(user_input):
    """
    Prepares user input, makes predictions, and returns a human-readable result and confidence score.
    """
    processed_input = preprocess_input(user_input)
    
    if processed_input is None:
        return "Invalid input. Please enter correct values.", None
    
    if processed_input.shape[1] != model.n_features_in_:
        return f"Feature mismatch error! Expected {model.n_features_in_}, but got {processed_input.shape[1]}.", None

    # Make prediction
    prediction = model.predict(processed_input)[0]
    
    # Get probability score (only if model supports it)
    probability = None
    if hasattr(model, "predict_proba"):
        probability = model.predict_proba(processed_input)[0][1]  # Probability of class 1

    return ("Subscribed" if prediction == 1 else "Not Subscribed"), probability

