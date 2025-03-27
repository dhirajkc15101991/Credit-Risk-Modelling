import json
import pickle
from typing import NoReturn

import pandas as pd
from pathlib import Path
import os
import json




def read_excel_data(path: str) -> pd.DataFrame:
    data = pd.read_excel(path)
    return data

def read_csv_data(path: str) -> pd.DataFrame:
    data = pd.read_csv(path)
    return data

# Merge the two dataframes, inner join so that no nulls are present
def merge_df(df1,df2):
     df = pd. merge ( df1, df2, how ='inner', left_on = ['PROSPECTID'], right_on = ['PROSPECTID'] )
     return df

#### Covert DF to csv and lload into target directory
def save_df_dir(df,dir):
    df=df.reset_index(drop=True)
    df.to_csv(dir, index=False)

#save feature list to json file
def save_features_to_json(features_list, json_file_path):
    """
    Saves a list of features into a JSON file.

    Parameters:
        features_list (list): List of feature names.
        dir_path (str or Path): Directory where the JSON file should be saved.
        filename (str): Name of the JSON file (default: 'features.json').

    Returns:
        None (Saves JSON file in the given directory)
    """

    with open(json_file_path, "w") as json_file:
        json.dump(features_list, json_file, indent=4)  # Save list as JSON


#load the save features 

def load_features_from_json(file_path):
    """
    Reads a list of features from a JSON file.

    Parameters:
        file_path (str or Path): Path to the JSON file.

    Returns:
        list: List of feature names.
    """
    file_path = Path(file_path)

    if not file_path.exists():
        print(f"Error: File {file_path} not found!")
        return []

    with open(file_path, "r") as json_file:
        features_list = json.load(json_file)  # Load JSON file

    return features_list

def load_pkl_file(input_: str):
    with open(input_, "rb") as fin:
        res = pickle.load(fin)
    return res

