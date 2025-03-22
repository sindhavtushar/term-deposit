import sys
import os
import pickle
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "utils")))

import utils_preprocess as validate

# check_categorical_features, check_class_imbalance, check_data_size, check_feature_scaling, check_missing_values, check_outliers

# print(dir(validate))

with open ('../preprocessed_data/preprocessed_data.pkl','rb') as f:
    X_train, X_test, y_train, y_test, _ = pickle.load(f)

print("Shape of preprocessed data:")
print("Training set: ",X_train.shape, y_train.shape)
print("Testing set: ",X_test.shape, y_test.shape)


# If X_train is a sparse matrix, convert it to a dense DataFrame
if hasattr(X_train, "toarray"):
    X_train = pd.DataFrame(X_train.toarray())

if hasattr(X_test, "toarray"):
    X_test = pd.DataFrame(X_test.toarray())

# Convert y_train to a DataFrame if it is a Series or numpy array
y_train = pd.DataFrame(y_train)
y_test = pd.DataFrame(y_test)

# Combine X_train and y_train into a temporary DataFrame for inspection
df_train = pd.concat([X_train, y_train], axis=1)

# Check the first few rows
print(df_train.head())

data = df_train

print('--------------------------------------------------------------------------------------------')
# Check for missing values
print('Missing Values Status:\n')
validate.check_missing_values(data)
print('--------------------------------------------------------------------------------------------')

print('--------------------------------------------------------------------------------------------')
print('Imbalance test of target feature:\n')
# Check for class imbalance in 'target'
validate.check_class_imbalance(data, 'y')
print('--------------------------------------------------------------------------------------------')

print('--------------------------------------------------------------------------------------------')
print('Categorical feature testing\n')
# Check for categorical features
validate.check_categorical_features(data)
print('--------------------------------------------------------------------------------------------')

print('--------------------------------------------------------------------------------------------')
print('Outliers testing in training set:\n')
# Check for outliers
validate.check_outliers(data)
print('--------------------------------------------------------------------------------------------')

print('--------------------------------------------------------------------------------------------')
print('checking the size of target feature:\n')
# Check data size for 'target' feature
validate.check_data_size(data, 'y')
print('--------------------------------------------------------------------------------------------')

print('--------------------------------------------------------------------------------------------')
print('Feature scalling of numerical features:\n')
# Check feature scaling for numerical columns
validate.check_feature_scaling(data)
print('--------------------------------------------------------------------------------------------')


