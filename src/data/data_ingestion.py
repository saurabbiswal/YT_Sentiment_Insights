import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
import yaml
import logging

# Logging configuration

# Create a logger for the data ingestion module
logger = logging.getLogger('data_ingestion')
logger.setLevel(logging.DEBUG)

# Create handlers for console and file logging
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

file_handler = logging.FileHandler('errors.log')
file_handler.setLevel(logging.ERROR)

# Create a formatter and set it for both handlers
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

# Adding handlers to the logger object
# logger has all the handlers attached to it with the specified levels and formatters. Ready to log messages.
logger.addHandler(console_handler)
logger.addHandler(file_handler)


# This function loads parameters from a YAML file and returns them as a dictionary.
def load_params(params_path: str) -> dict:
    """Load parameters from a YAML file."""
    try:
        with open(params_path, 'r') as file: # Open the YAML file - read mode
            params = yaml.safe_load(file)
        logger.debug('Parameters retrieved from %s', params_path)
        return params
    except FileNotFoundError:
        logger.error('File not found: %s', params_path)
        raise
    except yaml.YAMLError as e:
        logger.error('YAML error: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error: %s', e)
        raise

# This function loads data from a CSV file located at the specified URL and returns it as a pandas DataFrame.
# It handles errors related to file parsing and unexpected issues, logging them appropriately.
def load_data(data_url: str) -> pd.DataFrame:
    """Load data from a CSV file."""
    try:
        df = pd.read_csv(data_url)
        logger.debug('Data loaded from %s', data_url)
        return df
    except pd.errors.ParserError as e:
        logger.error('Failed to parse the CSV file: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error occurred while loading the data: %s', e)
        raise


# This function preprocesses the data by removing missing values, duplicates, and empty strings.
# It logs the completion of preprocessing and handles any errors related to missing columns or unexpected issues.
def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess the data by handling missing values, duplicates, and empty strings."""
    try:
        # Removing missing values
        df.dropna(inplace=True)
        # Removing duplicates
        df.drop_duplicates(inplace=True)
        # Removing rows with empty strings
        df = df[df['clean_comment'].str.strip() != '']
        
        logger.debug('Data preprocessing completed: Missing values, duplicates, and empty strings removed.')
        return df
    except KeyError as e:
        logger.error('Missing column in the dataframe: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error during preprocessing: %s', e)
        raise


# This function saves the train and test datasets to a specified path, creating a 'raw' folder if it doesn't exist.
# It logs the success of the operation and handles any errors related to file writing or unexpected issues 
def save_data(train_data: pd.DataFrame, test_data: pd.DataFrame, data_path: str) -> None:
    """Save the train and test datasets, creating the raw folder if it doesn't exist."""
    try:
        raw_data_path = os.path.join(data_path, 'raw')
        
        # Create the data/raw directory if it does not exist
        os.makedirs(raw_data_path, exist_ok=True)
        
        # Save the train and test data
        train_data.to_csv(os.path.join(raw_data_path, "train.csv"), index=False)
        test_data.to_csv(os.path.join(raw_data_path, "test.csv"), index=False)
        
        logger.debug('Train and test data saved to %s', raw_data_path)
    except Exception as e:
        logger.error('Unexpected error occurred while saving the data: %s', e)
        raise


# Main function to orchestrate the data ingestion process
# It loads parameters, fetches data, preprocesses it, splits it into train and test sets, and saves the datasets.
# It also handles exceptions and logs errors appropriately.
def main():
    try:
        # Load parameters from the params.yaml in the root directory
        params = load_params(params_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../params.yaml'))
        test_size = params['data_ingestion']['test_size']
        
        # Load data from the specified URL
        df = load_data(data_url='https://raw.githubusercontent.com/Himanshu-1703/reddit-sentiment-analysis/refs/heads/main/data/reddit.csv')
        
        # Preprocess the data
        final_df = preprocess_data(df)
        
        # Split the data into training and testing sets
        train_data, test_data = train_test_split(final_df, test_size=test_size, random_state=42)
        
        # Save the split datasets and create the raw folder if it doesn't exist
        save_data(train_data, test_data, data_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../data'))
        
    except Exception as e:
        logger.error('Failed to complete the data ingestion process: %s', e)
        print(f"Error: {e}")

if __name__ == '__main__':
    main()
