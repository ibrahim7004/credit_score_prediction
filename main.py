import pandas as pd
import numpy as np
import re
import os
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor


# Generalised function to handle and process all differnt data in the columns:
def process_data(column):
    processed_values = []

    for value in column:
        if pd.isna(value):  # Handle empty or NaN values
            processed_values.append(None)
            continue
        
        # Extract numeric values using regular expression
        numbers = re.findall(r'\d+', value)
        
        # Handle cases where multiple numeric values are found
        if len(numbers) >= 2:
            start, end = map(int, numbers)
            if 'k' in value.lower(): 
                start *= 1000
                end *= 1000
            avg = (start + end) / 2
            processed_values.append(int(avg))
        # Handle cases where only one numeric value is found
        elif len(numbers) == 1:
            num = int(numbers[0])
            if 'k' in value.lower():  
                num *= 1000
            processed_values.append(num)
        else:
            # Handle cases where numeric values are not directly present
            if 'to' in value.lower() or '-' in value:
                numbers = re.findall(r'\d+', value)
                start, end = map(int, numbers)
                if 'k' in value.lower():  
                    start *= 1000
                    end *= 1000
                avg = (start + end) / 2
                processed_values.append(int(avg))
            elif '+' in value:
                num = int(value.replace('+', ''))
                if 'k' in value.lower():  
                    num *= 1000
                processed_values.append(num)
            elif 'more' in value.lower():
                num = int(numbers[0])
                processed_values.append(num)
            elif 'less' in value.lower():
                num = int(numbers[0])
                processed_values.append(num)
            else:
                processed_values.append(None)
    
    return processed_values


# Function to drop a specific column
def drop_column(data_frame, column_name):
    new_df = data_frame.drop(columns=[column_name])
    return new_df


def preprocess(df):
    # Convert 'total_sales' column to numerical
    df['total_sales'] = df['total_sales'].str.replace(',', '').str.strip()
    df['total_sales'] = pd.to_numeric(df['total_sales'], errors='coerce')
    
    # Apply the process_data function to multiple columns
    columns_to_process = ['aprox_exist_inventory', 'no_of_products', 'number_of_orders', 
                        'avg_daily_sales', 'rent_amount', 'gmv', 'using_pos', 'shop_size', 
                        'business_age(year)', 'electricity_bill']

    for column_name in columns_to_process:
        df[column_name] = process_data(df[column_name])

    # Call the function to drop the 'using_pos' column
    new_df = drop_column(df, 'using_pos')

    return new_df


def encoding(df):
    # Initialize the LabelEncoder
    label_encoder = LabelEncoder()

    # Fit and transform the 'shop_type' and 'is_rental' columns using label encoding:
    df['shop_type'] = label_encoder.fit_transform(df['shop_type'])
    df['is_rental'] = label_encoder.fit_transform(df['is_rental'])

    return df


def remove_rows_with_nan(data_frame):
    cleaned_data = data_frame.dropna()  # Drop all rows with missing values
    return cleaned_data


def normalization(df, numerical_features):
    # Apply Standardization to numerical features
    scaler = StandardScaler()
    df_normalized = df.copy()  # Create a copy to avoid modifying the original DataFrame
    df_normalized[numerical_features] = scaler.fit_transform(df_normalized[numerical_features])

    return df_normalized


def perform_data_split(df, test_size=0.2, random_state=None):   # Setting test size to 20% and training to 80%.
    X = df.drop(columns=['credit_score'])
    y = df['credit_score']
    
    # Split dataset into training and testing sets using train_test_split function
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    # Return training and testing feature matrices and target vectors
    return X_train, X_test, y_train, y_test


# Function to train a Linear Regression model
def train_linear_regression(X_train, y_train, X_test, y_test):
    # Create a Linear Regression model
    model = LinearRegression()
    # Train the model using training data
    model.fit(X_train, y_train)
    # Make predictions on test data
    y_pred = model.predict(X_test)
    
    # Return true labels, predicted labels, and model name
    return y_test, y_pred, "Linear Regression"

# Function to train a Random Forest model
def train_random_forest(X_train, y_train, X_test, y_test):
    # Create a Random Forest model
    model = RandomForestRegressor()
    # Train the model using training data
    model.fit(X_train, y_train)
    # Make predictions on test data
    y_pred = model.predict(X_test)
    
    # Return true labels, predicted labels, and model name
    return y_test, y_pred, "Random Forest"

# Function to train an XGBoost model
def train_xgboost(X_train, y_train, X_test, y_test):
    # Create an XGBoost model
    model = XGBRegressor()
    # Train the model using training data
    model.fit(X_train, y_train)
    # Make predictions on test data
    y_pred = model.predict(X_test)
    
    # Return true labels, predicted labels, and model name
    return y_test, y_pred, "XGBoost"

# Function to train a Neural Network model
def train_neural_network(X_train, y_train, X_test, y_test):
    # Create a Neural Network model
    model = MLPRegressor()
    # Train the model using training data
    model.fit(X_train, y_train)
    # Make predictions on test data
    y_pred = model.predict(X_test)
    
    # Return true labels, predicted labels, and model name
    return y_test, y_pred, "Neural Network"

# Function to train a Support Vector Machine (SVM) model
def train_svm(X_train, y_train, X_test, y_test):
    # Create an SVM model
    model = SVR()
    # Train the model using training data
    model.fit(X_train, y_train)
    # Make predictions on test data
    y_pred = model.predict(X_test)
    
    # Return true labels, predicted labels, and model name
    return y_test, y_pred, "Support Vector Machine"

# Function to train a K-Nearest Neighbors (KNN) model
def train_knn(X_train, y_train, X_test, y_test):
    # Create a KNN model
    model = KNeighborsRegressor()
    # Train the model using training data
    model.fit(X_train, y_train)
    # Make predictions on test data
    y_pred = model.predict(X_test)
    
    # Return true labels, predicted labels, and model name
    return y_test, y_pred, "K-Nearest Neighbours"


def evaluate_metrics(y_test, y_pred, model_name):
    model_metrics = []
    n_features = 12     # No. of features being used

    # Calculate the Mean Squared Error (MSE) between true labels (y_test) and predicted labels (y_pred)
    mse = mean_squared_error(y_test, y_pred)
    
    # Calculate the Root Mean Squared Error (RMSE) by taking the square root of the MSE
    rmse = np.sqrt(mse)
    
    # Calculate the Mean Absolute Error (MAE) between true labels (y_test) and predicted labels (y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    
    # Calculate the R-squared (coefficient of determination) score between true labels (y_test) and predicted labels (y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Calculate the Adjusted R-squared score, which adjusts R-squared based on the number of features used in the model
    # It penalizes overfitting and takes into account the number of samples (len(y_test)) and number of features (n_features)
    adjusted_r2 = 1 - ((1 - r2) * (len(y_test) - 1) / (len(y_test) - n_features - 1))

    
    # Append all metrics to the model_metrics list
    model_metrics.append(model_name)
    model_metrics.append(mse)
    model_metrics.append(rmse)
    model_metrics.append(mae)
    model_metrics.append(r2)
    model_metrics.append(adjusted_r2)
    
    return model_metrics

def execute_models(df):
    functions_to_execute = [train_linear_regression, train_random_forest, train_xgboost, train_neural_network, train_svm, train_knn]
    metrics = []

    # Loop through each model function and execute it
    for func in functions_to_execute:
        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = perform_data_split(df)
        
        # Train the model, make predictions, and get model name
        y_true, y_pred, name = func(X_train, y_train, X_test, y_test)
        
        # Calculate evaluation metrics for the model's predictions
        model_metrics = evaluate_metrics(y_true, y_pred, name)
        
        # Append the model's metrics (excluding the name) to the list
        metrics.append([name] + model_metrics[1:])

    create_df(metrics)


def create_df(model_metrics):
    # Define column names for the DataFrame
    columns = ['Model', 'MSE', 'RMSE', 'MAE', 'R-2', 'Adjusted R-2']
    
    # Create a new DataFrame using the metrics data and column names
    new_df = pd.DataFrame(model_metrics, columns=columns)

    print("\n\n\n", new_df, "\n\n")


def initial_tasks(directory):
    # Read the CSV file
    df = pd.read_csv(os.path.join(directory, 'CreditScoring.csv'))

    # List of numerical features for scaling
    numerical_features = ['number_of_orders', 'no_of_products', 'total_sales', 'gmv',
        'avg_daily_sales', 'aprox_exist_inventory', 'shop_size',
        'business_age(year)', 'electricity_bill', 'rent_amount']
    
    return df, numerical_features


def main():
    
    # Set your directory:
    existing_directory = r'C:\Users\hp\Desktop\codeNinja\week8'

    df, numerical_features = initial_tasks(existing_directory)

    # Preprocessing steps:
    processed_df = preprocess(df)
    encoded_df = encoding(processed_df)
    preprocessed_df = remove_rows_with_nan(encoded_df)
    normalized_df = normalization(preprocessed_df, numerical_features)

    # Storing fully pre-processed dataframe to a new CSV:
    normalized_df.to_csv(os.path.join(existing_directory, 'output.csv'), index=False)

    # Splitting data, training models, making predictions, and evaluating model accuracy:
    execute_models(normalized_df)


main()