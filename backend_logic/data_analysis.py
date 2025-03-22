import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
filePath = "data/bank-full.csv"
df = pd.read_csv(filePath, sep=";")

# 1. Check for missing values
def check_missing_values(data):
    missing = data.isnull().sum()
    print("Missing Values:")
    print(missing[missing > 0] if missing.any() else "No missing values.")

# 2. Outlier Detection using IQR Method
def detect_outliers(data, numerical_features):
    print("\nOutlier Detection using IQR:")
    outlier_counts = {}
    for feature in numerical_features:
        Q1 = data[feature].quantile(0.25)
        Q3 = data[feature].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = data[(data[feature] < lower_bound) | (data[feature] > upper_bound)]
        outlier_counts[feature] = len(outliers)
        print(f"{feature}: {len(outliers)} outliers detected")
    
    # Boxplots for visualization
    plt.figure(figsize=(12, 6))
    data[numerical_features].boxplot()
    plt.title("Boxplot of Numerical Features")
    plt.xticks(rotation=45)
    plt.show()

# 3. Target Class Imbalance
def check_class_imbalance(data, target):
    class_counts = data[target].value_counts(normalize=True) * 100
    print("\nTarget Class Distribution:")
    print(class_counts)

    # Update countplot to avoid FutureWarning: Assign hue to the target variable
    sns.countplot(x=target, data=data, hue=target, palette="viridis", legend=False)
    plt.title("Target Variable Distribution")
    plt.show()

# 4. Feature Distribution Analysis
def plot_feature_distributions(data, numerical_features):
    data[numerical_features].hist(figsize=(12, 8), bins=20, color='blue', alpha=0.7)
    plt.suptitle("Feature Distributions", fontsize=16)
    plt.show()

# 5. Correlation Analysis
def correlation_analysis(data, target):
    # Select only numerical columns for correlation
    numerical_data = data.select_dtypes(include=[np.number])
    correlation_matrix = numerical_data.corr()  # Use only numerical data

    # Plot heatmap of correlations
    plt.figure(figsize=(10, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Feature Correlation Matrix")
    plt.show()

    print("\nCorrelation with target variable:")
    print(correlation_matrix[target].sort_values(ascending=False))

# 6. Categorical Feature Analysis
def analyze_categorical_features(data, categorical_features):
    for feature in categorical_features:
        plt.figure(figsize=(8, 4))
        sns.countplot(x=feature, data=data, hue=feature, palette="pastel", order=data[feature].value_counts().index, legend=False)
        plt.title(f"Distribution of {feature}")
        plt.xticks(rotation=45)
        plt.show()

# Define feature types
numerical_features = ["age", "balance", "day", "duration", "campaign", "pdays", "previous"]
categorical_features = ["job", "marital", "education", "default", "housing", "loan", "contact", "month", "poutcome"]
target_variable = "y"

# Convert target variable to binary (yes=1, no=0)
df[target_variable] = df[target_variable].map({"yes": 1, "no": 0})

# Run analyses
check_missing_values(df)
detect_outliers(df, numerical_features)
check_class_imbalance(df, target_variable)
plot_feature_distributions(df, numerical_features)
correlation_analysis(df, target_variable)
analyze_categorical_features(df, categorical_features)
