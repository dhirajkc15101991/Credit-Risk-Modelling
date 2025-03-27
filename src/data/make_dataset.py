from src.entities import SplittingParams

from sklearn.model_selection import train_test_split

def split_data(df, train_size=SplittingParams.train_size, val_size=SplittingParams.val_size, random_state=SplittingParams.random_state):
    """
    Splits a DataFrame into training and validation sets.
    
    Parameters:
        df (pd.DataFrame): The full dataset.
        train_size (float): Proportion of the dataset for training (default: 80%).
        val_size (float): Proportion for validation (default: 20%).
        random_state (int): Random seed for reproducibility.

    Returns:
        df_train (pd.DataFrame): Training dataset.
        df_validation (pd.DataFrame): Validation dataset.
    """
    
    # Ensure the split sizes sum to 1
    assert train_size + val_size == 1, "train and validation sizes must sum to 1"
    
    # Split into train and validation
    df_train, df_validation = train_test_split(df, test_size=val_size, random_state=random_state)
    
    return df_train, df_validation
