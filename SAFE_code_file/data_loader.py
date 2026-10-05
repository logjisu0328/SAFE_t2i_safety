import pandas as pd
import os

def load_data(data_path):
    """
    Load data from CSV file.
    
    Expected data format:
    - prompt: Actual prompt text (str)
    - ground_truth: Target level per category as dict stored in string format
      Example: "{'O2: Sexual Content': 1, 'O3: Violence': 2}"
    - culture: Target culture/country for cultural awareness (str, optional)
      Required only when using --culture on
      Example: "Korea", "Saudi Arabia", "Singapore"
    
    Args:
        data_path (str): Path to CSV file
    
    Returns:
        pd.DataFrame: Loaded dataframe
    """
    print(f"[INFO] Loading data from {data_path}")
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    df = pd.read_csv(data_path)
    
    # Check required columns
    required_columns = ['prompt', 'ground_truth']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    print(f"[INFO] Loaded {len(df)} samples")
    print(f"[INFO] Columns: {list(df.columns)}")
    
    return df