import joblib
import numpy as np
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from load_models import load_data, load_models  # Import utility functions

# Load data & models
X_train, X_test, y_train, y_test, _ = load_data()
models = load_models()

# Convert sparse matrices if needed
X_train = X_train.toarray() if hasattr(X_train, "toarray") else X_train
X_test = X_test.toarray() if hasattr(X_test, "toarray") else X_test

# Ensemble - Soft Voting
ensemble_model = VotingClassifier(estimators=list(models.items()), voting="soft")

# Cross-validation on the ensemble model
cv_scores = cross_val_score(ensemble_model, X_train, y_train, cv=5, scoring="accuracy")  # 5-fold cross-validation

# Print cross-validation results
print(f"Cross-validation accuracy scores: {cv_scores}")
print(f"Mean cross-validation accuracy: {cv_scores.mean():.4f}")
print(f"Standard deviation of cross-validation accuracy: {cv_scores.std():.4f}")

# Fit the ensemble model on the full training data (needed before final prediction)
ensemble_model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = ensemble_model.predict(X_test)

# Print accuracy of each individual model
print("\nIndividual Model Accuracies:")
for name, model in models.items():
    y_pred_individual = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred_individual)
    print(f"{name} Accuracy: {accuracy:.4f}")

# Print accuracy of ensemble model
ensemble_accuracy = accuracy_score(y_test, y_pred)
print(f"\nEnsemble Model Accuracy: {ensemble_accuracy:.4f}")
