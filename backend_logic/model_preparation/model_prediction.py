import joblib
import numpy as np
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score
from load_models import load_data, load_models  # Import utility functions

# Load data & models
X_train, X_test, y_train, y_test, _ = load_data()
models = load_models()

# Convert sparse matrices if needed
X_train = X_train.toarray() if hasattr(X_train, "toarray") else X_train
X_test = X_test.toarray() if hasattr(X_test, "toarray") else X_test

# Ensemble - Soft Voting
ensemble_model = VotingClassifier(estimators=list(models.items()), voting="soft")
ensemble_model.fit(X_train, y_train)

# Save the ensemble model
joblib.dump(ensemble_model, "trained_model/ensemble_model.pkl")
print("✅ Ensemble model saved as 'ensemble_model.pkl'")

# Function to evaluate model performance
def evaluate_model(model, name):
    """Evaluate a model and print its performance metrics."""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1 Score": f1_score(y_test, y_pred),
        "ROC AUC": roc_auc_score(y_test, y_prob) if y_prob is not None else "N/A",
        "PR AUC": average_precision_score(y_test, y_prob) if y_prob is not None else "N/A",
    }

    print(f"\n🔍 {name} Performance:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}" if value != "N/A" else f"{metric}: N/A")

# Evaluate individual models
for name, model in models.items():
    evaluate_model(model, name)

# Evaluate ensemble model
evaluate_model(ensemble_model, "Ensemble Model")
