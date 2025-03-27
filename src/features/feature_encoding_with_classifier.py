from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder,OrdinalEncoder,StandardScaler,LabelEncoder
import joblib
import pandas as pd
import os

def create_save_feature_encoder_classifier(features_list,df_training_cust,file_path):
    # select onnly features that we willl work on
    df_training_cust_work_upon=df_training_cust[features_list]
    # Identify numerical columns
    numerical_cols_training_cust = df_training_cust_work_upon.select_dtypes(include=['int64', 'float64']).columns.tolist()
    # Identify categorical columns
    categorical_cols_training_cust = df_training_cust_work_upon.select_dtypes(include=['object', 'category']).columns.tolist()
    # print(categorical_cols_training_cust)
    #ordinal columns
    categorical_cols_training_ordinal = ['EDUCATION']
    categorical_cols_training_cust_non_ordinal = [col for col in categorical_cols_training_cust if col not in categorical_cols_training_ordinal]
    # Define ordinal encoding categories for EDUCATION
    education_categories = [['SSC', '12TH', 'UNDER GRADUATE', 'GRADUATE', 'OTHERS', 'POST-GRADUATE', 'PROFESSIONAL']]

    # Create preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('ordinal', OrdinalEncoder(categories=education_categories), ['EDUCATION']),  # Ordinal encoding for EDUCATION
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse=False),categorical_cols_training_cust_non_ordinal),  # One-hot encoding
            ('scaler', StandardScaler(), numerical_cols_training_cust)  # Standardize numerical columns
        ]
    )
    preprocessor.fit(df_training_cust_work_upon)
    
    joblib.dump(preprocessor,file_path)

# save label Encoder classifier
def save_lable_encoder(file_name,y_train):
    label_encoder_target=LabelEncoder()
    label_encoder_target.fit(y_train)
    joblib.dump(label_encoder_target,file_name)