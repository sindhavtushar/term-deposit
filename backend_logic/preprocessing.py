import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# Load the dataset
filePath = "data/bank-full.csv"
df = pd.read_csv(filePath, sep=";")
# Define feature types
numerical_features = ["age", "balance", "day", "duration", "campaign", "pdays", "previous"]
categorical_features = ["job", "marital", "education", "default", "housing", "loan", "contact", "month", "poutcome"]
target_variable = "y"

# Convert target variable to binary (yes=1, no=0)
df[target_variable] = df[target_variable].map({"yes": 1, "no": 0})

# Define preprocessing steps
preprocessor = ColumnTransformer([
    ("num_scaling", StandardScaler(), numerical_features),  # Scale numerical features
    ("cat_encoding", OneHotEncoder(handle_unknown="ignore", drop="first"), categorical_features)  # Encode categorical features
])

# Apply transformations
X = df.drop(columns=[target_variable])
y = df[target_variable]

X_processed = preprocessor.fit_transform(X)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42, stratify=y)

# Save preprocessed data
with open("preprocessed_data/preprocessed_data.pkl", "wb") as f:
    pickle.dump((X_train, X_test, y_train, y_test, preprocessor), f)

print("Preprocessing complete. Data saved as 'preprocessed_data.pkl'.")
