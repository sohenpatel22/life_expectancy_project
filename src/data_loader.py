import os
import pandas as pd
from sklearn.model_selection import train_test_split

def load_and_split_data(url, local_path='../data/LifeExpectancy.csv', test_size=0.3, random_state=0):
    """
    Downloads data if it doesn't exist locally, loads it, 
    creates the binary target, and splits into train/test.
    """
    # 1. Check if we already downloaded the data (Caching)
    if not os.path.exists(local_path):
        df = pd.read_csv(url)
        
        # Create the 'data/' folder if it doesn't exist, then save the CSV
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        df.to_csv(local_path, index=False)
        print("Dataset cached locally.")
    else:
        print("Loading dataset from local cache")
        df = pd.read_csv(local_path)
    
    # 2. Create the binary target based on the median
    median_val = df['Life expectancy '].median()
    df['target_binary'] = (df['Life expectancy '] >= median_val).astype(int)
    
    # 3. Drop the original target and new target from features
    X = df.drop(columns=['Life expectancy ', 'target_binary'])
    y = df['target_binary']
    
    # 4. Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    return X_train, X_test, y_train, y_test