import joblib
import pickle
import os
import pandas as pd
import scipy.sparse
from collections import Counter
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, GridSearchCV

# Ensure the directory exists for saving models
os.makedirs("trained_model", exist_ok=True)

# Load preprocessed data
filePath = "../preprocessed_data/preprocessed_data.pkl"

def load_data():
    """Load preprocessed data from pickle file."""
    with open(filePath, "rb") as f:
        return pickle.load(f)

# Unpacking the data
X_train, X_test, y_train, y_test, _ = load_data()

# Convert sparse matrices if needed
if scipy.sparse.issparse(X_train):
    X_train = X_train.toarray()
    X_test = X_test.toarray()

# First split the data
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Apply SMOTE on the training data only
X_train_resampled, y_train_resampled = SMOTE().fit_resample(X_train, y_train)

# Compute scale_pos_weight dynamically for XGBoost
class_counts = Counter(y_train_resampled)
scale_pos_weight = class_counts[0] / class_counts[1]

# Train Models
print("Training Models...")

# 1. XGBoost
xgb_model = XGBClassifier(scale_pos_weight=scale_pos_weight)
xgb_model.fit(X_train_resampled, y_train_resampled)
joblib.dump(xgb_model, "trained_model/xgb_model.pkl")

# 2. Random Forest
rf_model = RandomForestClassifier(class_weight="balanced", n_estimators=100)
rf_model.fit(X_train_resampled, y_train_resampled)
joblib.dump(rf_model, "trained_model/rf_model.pkl")

# 3. GaussianNB with GridSearch
param_grid = {"var_smoothing": [1e-9, 1e-8, 1e-7]}
grid_nb = GridSearchCV(GaussianNB(), param_grid, cv=5)
grid_nb.fit(X_train_resampled, y_train_resampled)
joblib.dump(grid_nb.best_estimator_, "trained_model/nb_model.pkl")

# 4. KNN
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train_resampled, y_train_resampled)
joblib.dump(knn_model, "trained_model/knn_model.pkl")

print("Model Training Complete & Saved!")
