import os
import logging.config
from typing import Dict

from omegaconf import DictConfig,OmegaConf
import pandas as pd
import hydra

from src.entities.train_pipeline_params import TrainingPipelineParams, TrainingPipelineParamsSchema
from src.data import split_data
from src.features.handling_null_value import null_removal_df_cibil_data_cust,null_removal_df_internal_data_cust
from src.features.feature_selection import feature_selection_data
from src.features.feature_encoding_with_classifier import create_save_feature_encoder_classifier,save_lable_encoder
from src.utils import read_excel_data,read_csv_data,merge_df,save_df_dir,save_features_to_json,load_features_from_json
from src.models.train_model import training_model
from pathlib import Path
import joblib

logger = logging.getLogger("train_pipeline")


def train_pipeline(
        training_pipeline_params: TrainingPipelineParams,
) -> Dict[str, float]:
    logger.info(f"Start train pipeline with params {training_pipeline_params}")
    logger.info(f"Model is {training_pipeline_params.train_params.model_type}")

    BASE_DIR = Path(os.path.dirname(os.path.abspath(__file__))).parent/'Credit-Risk-Modelling'
    # print(BASE_DIR)
    # # print(BASE_DIR)
    # # print(Path(training_pipeline_params.path_config.raw_internal_data_path))
    # print(BASE_DIR/Path(training_pipeline_params.path_config.raw_internal_data_path))

    # read the raw data from data/raw_Data -Source data for our training
    logger.info(f"Started loading the  raew file file - {training_pipeline_params.path_config.raw_internal_data_path} and {training_pipeline_params.path_config.raw_cibil_data_path} ")
    internal_data_cust=read_excel_data(BASE_DIR/Path(training_pipeline_params.path_config.raw_internal_data_path))
    cibil_data_cust=read_excel_data(BASE_DIR/Path(training_pipeline_params.path_config.raw_cibil_data_path))
    logger.info(f"Completed loading the  raew file file - {training_pipeline_params.path_config.raw_internal_data_path} and {training_pipeline_params.path_config.raw_cibil_data_path} ")
    logger.info(f"shape of {training_pipeline_params.path_config.raw_internal_data_path} before null removal --->{internal_data_cust.shape} ")
    logger.info(f"shape of {training_pipeline_params.path_config.raw_cibil_data_path} before null removal --->{cibil_data_cust.shape} ")
    #Null Value removal
    df_internal_data_cust=null_removal_df_internal_data_cust(internal_data_cust)
    df_cibil_data_cust=null_removal_df_cibil_data_cust(cibil_data_cust)
    logger.info(f"shape of {training_pipeline_params.path_config.raw_internal_data_path} before null removal --->{df_internal_data_cust.shape} ")
    logger.info(f"shape of {training_pipeline_params.path_config.raw_cibil_data_path} before null removal --->{df_cibil_data_cust.shape} ")
    #Merge two dataframe based on custid or Prospects_id
    logger.info("Merging Raw Internal data and Raw Cibil Data")
    df_final_data_cust=merge_df(df_internal_data_cust,df_cibil_data_cust)
    logger.info(f"Shape after Merging two raw Data :{df_final_data_cust.shape}")
    #Splitting the data into tarining and validation
    logger.info(f"Training and validation datasets splitting started")
    df_train,df_validation=split_data(df_final_data_cust,train_size=training_pipeline_params.splitting_params.train_size, val_size=training_pipeline_params.splitting_params.val_size, random_state=training_pipeline_params.splitting_params.random_state)
    logger.info(f" Initial level Training Datasets shape-Before feature selection-{df_train.shape} and Initial Label Validation/Testing Data -Before Feature Selection shape-{df_validation.shape}")
    #Stroring the Tarining and Validation Data in traininga nd validation Data dir inside data folder
    train_data_file=BASE_DIR/Path(training_pipeline_params.path_config.train_data_path)
    validation_data_file=BASE_DIR/Path(training_pipeline_params.path_config.val_data_path)
    logger.info(f"Saving Training {train_data_file} and validation {validation_data_file} data file ---Started")
    save_df_dir(df_train,train_data_file)
    save_df_dir(df_validation,validation_data_file)
    logger.info(f"Saving Training {train_data_file} and validation {validation_data_file} data files---Completed")
    logger.info(f"Loading the Tarining Data file for feature selection-->{train_data_file}")
    df_training_cust=read_csv_data(train_data_file)
    logger.info(f"starting feature selection for {train_data_file}")
    logger.info(f"Total no of Features before feature selection--->{len(df_training_cust.columns)-1}")
    selected_features_list_without_target=feature_selection_data(df_training_cust,target_column=training_pipeline_params.feature_params.target_col)
    logger.info(f"Total no of Features afetr feature selection--->{len(selected_features_list_without_target)}")
    logger.info(f"Storing the final features in {training_pipeline_params.path_config.feature_store_path} started")
    save_features_to_json(selected_features_list_without_target,BASE_DIR/Path(training_pipeline_params.path_config.feature_store_path))
    logger.info(f"Storing the final features in {training_pipeline_params.path_config.feature_store_path} completed")
    logger.info(f"Loading the final features from {training_pipeline_params.path_config.feature_store_path} started")
    features_list_loaded=load_features_from_json(BASE_DIR/Path(training_pipeline_params.path_config.feature_store_path))
    logger.info(f"Loading the final features from {training_pipeline_params.path_config.feature_store_path} Completed")
    print(features_list_loaded)
    logger.info(f"Feature encoding and saving its classifier to {BASE_DIR/training_pipeline_params.path_config.feature_encoded_classifier_path} started")
    create_save_feature_encoder_classifier(features_list_loaded,df_training_cust,BASE_DIR/training_pipeline_params.path_config.feature_encoded_classifier_path)
    logger.info(f"Feature encoding and saving its classifier to {BASE_DIR/training_pipeline_params.path_config.feature_encoded_classifier_path} Completed")
    logger.info(f"Label encoding saving  to its classifier {BASE_DIR/training_pipeline_params.path_config.label_encoder_classifier_path} started")
    save_lable_encoder(BASE_DIR/training_pipeline_params.path_config.label_encoder_classifier_path,df_training_cust[training_pipeline_params.feature_params.target_col])
    logger.info(f"Label encoding saving  to its classifier {BASE_DIR/training_pipeline_params.path_config.label_encoder_classifier_path} completed")
    #train and save the model
    label_encoder_risk_modelling=joblib.load(BASE_DIR/training_pipeline_params.path_config.label_encoder_classifier_path)
    preprocessor_risk_model=joblib.load(BASE_DIR/training_pipeline_params.path_config.feature_encoded_classifier_path)
    X_train=df_training_cust[features_list_loaded]
    y_train=df_training_cust[training_pipeline_params.feature_params.target_col]
    model_name=training_pipeline_params.train_params.model_type #RandomForestClassifier
    param_distributions=training_pipeline_params.train_params.param_grid
    save_dir_path=BASE_DIR/training_pipeline_params.path_config.output_model_path
    experiment_dir=BASE_DIR/training_pipeline_params.path_config.experiment_dir
    logger.info("Started training of {model_name} models")
    training_model(param_distributions,preprocessor_risk_model,label_encoder_risk_modelling,X_train,y_train,save_dir_path,model_name,experiment_dir)
    logger.info(f"Completed training of {model_name} models and metric is stored in {experiment_dir}")
    
    return


@hydra.main(config_path="../configs", config_name="train_config",version_base="1.1")
def train_pipeline_start(cfg: DictConfig):
    os.chdir(hydra.utils.to_absolute_path(".."))
    schema = TrainingPipelineParamsSchema()
    params = schema.load(cfg)
    train_pipeline(params)


if __name__ == "__main__":
    train_pipeline_start()
