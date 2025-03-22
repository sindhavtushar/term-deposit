# Function used to preprocess or validate data

def check_missing_values(data):
    missing_values = data.isnull().sum()
    missing_columns = missing_values[missing_values > 0]
    
    if missing_columns.empty:
        print("No missing values in the dataset.")
    else:
        print("Missing values found in the following columns:")
        print(missing_columns)


def check_class_imbalance(data, target_feature):
    class_counts = data[target_feature].value_counts()
    print(f"Class distribution for '{target_feature}':\n", class_counts)
    
    total_samples = len(data)
    imbalance_ratio = class_counts.min() / class_counts.max()
    
    if imbalance_ratio < 0.2:  # Arbitrary threshold for imbalance
        print("\nWarning: The dataset may be highly imbalanced.")
    else:
        print("\nClass distribution seems balanced.")


def check_categorical_features(data):
    categorical_columns = data.select_dtypes(include=['object', 'category']).columns
    print("Categorical columns in the dataset:")
    print(categorical_columns)
    
    # Suggest encoding methods
    for column in categorical_columns:
        print(f"\nFor column '{column}', consider encoding methods like:")
        print(" - One-hot encoding (for nominal categories)")
        print(" - Label encoding (for ordinal categories)")


def check_outliers(data):
    numerical_columns = data.select_dtypes(include=['number']).columns
    for column in numerical_columns:
        Q1 = data[column].quantile(0.25)
        Q3 = data[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = data[(data[column] < lower_bound) | (data[column] > upper_bound)]
        
        if not outliers.empty:
            print(f"\nOutliers detected in column '{column}':")
            print(outliers[column].head())
        else:
            print(f"\nNo outliers detected in column '{column}'.")


def check_data_size(data, target_feature, min_samples_per_class=50):
    class_counts = data[target_feature].value_counts()
    
    insufficient_samples = class_counts[class_counts < min_samples_per_class]
    
    if not insufficient_samples.empty:
        print("\nInsufficient samples in the following classes:")
        print(insufficient_samples)
    else:
        print("\nDataset has a sufficient number of samples for each class.")


def check_feature_scaling(data):
    numerical_columns = data.select_dtypes(include=['number']).columns
    for column in numerical_columns:
        min_value = data[column].min()
        max_value = data[column].max()
        range_diff = max_value - min_value
        
        if range_diff > 100:  # Arbitrary threshold to detect large range differences
            print(f"\nColumn '{column}' has a wide range of values (min: {min_value}, max: {max_value}). Consider normalizing.")
        else:
            print(f"\nColumn '{column}' has a reasonable range.")


def check_data_types(df):
    """Check if numerical and categorical features have the correct data types."""
    print("🔍 Checking data types...")
    print(df.dtypes)


def check_duplicates(df):
    """Check for duplicate rows in the dataset."""
    duplicates = df.duplicated().sum()
    if duplicates == 0:
        print("✅ No duplicate rows.")
    else:
        print(f"❌ Warning: {duplicates} duplicate rows found!")
        print(df[df.duplicated()])

def check_feature_distribution(df):
    """Check feature distributions using summary statistics."""
    print("🔍 Feature Distribution Summary:")
    print(df.describe())

def check_train_test_split(df, target_column, test_size=0.2):
    """Verify train-test split if not already done."""
    X_train, X_test, y_train, y_test = train_test_split(df.drop(columns=[target_column]), df[target_column], test_size=test_size, random_state=42)
    print(f"✅ Train set: {X_train.shape[0]} samples, Test set: {X_test.shape[0]} samples")
    return X_train, X_test, y_train, y_test

