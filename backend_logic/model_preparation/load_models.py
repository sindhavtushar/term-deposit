import joblib
import pickle
import numpy as np

# Load preprocessed data
filePath = "../preprocessed_data/preprocessed_data.pkl"

def load_data():
    """Load preprocessed data from pickle file."""
    with open(filePath, "rb") as f:
        return pickle.load(f)

def load_models():
    """Load trained models and return them as a dictionary."""
    models = {
        "XGB": joblib.load("trained_model/xgb_model.pkl"),
        "RF": joblib.load("trained_model/rf_model.pkl"),
        "NB": joblib.load("trained_model/nb_model.pkl"),
        "KNN": joblib.load("trained_model/knn_model.pkl"),
    }
    return models
