# Importing the required libraries
import pandas as pd
import os

def get_dataset(dataset: str):
    match dataset:
        case "IRIS":
            # Download the data if not available
            # URL = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'

            # Fetching the data from the local storage
            URL =  os.path.join(os.getenv("DATASETS"), 'IRIS/iris.csv')
            data = pd.read_csv(URL, header = None, encoding='utf-8')
            return data
        
        case _:
            raise ValueError("Invalid dataset")