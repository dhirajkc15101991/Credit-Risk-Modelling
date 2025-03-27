import os
import logging.config

import pandas as pd
from omegaconf import DictConfig
import hydra
import joblib
import json

from src.entities.predict_pipeline_params import PredictingPipelineParams,PredictingPipelineParamsSchema
from src.models.predict_model import validate_model
from src.utils import load_features_from_json

from pathlib import Path

logger = logging.getLogger("predict_pipeline")


def predict_pipeline(evaluating_pipeline_params: PredictingPipelineParams):
    logger.info(f"Start prediction pipeline {evaluating_pipeline_params}")
    BASE_DIR = Path(os.path.dirname(os.path.abspath(__file__))).parent/'Credit-Risk-Modelling'
    validation_data_file=BASE_DIR/Path(evaluating_pipeline_params.validation_data_path)
    logger.info(f"Laoding validation data {validation_data_file} started")
    df_validation=pd.read_csv(validation_data_file)
    logger.info(f"Laoding validation data {validation_data_file} completed")
    logger.info(f"loading classifiers - preprocessing,label encoding and important features started")
    preprocessor_file_path=BASE_DIR/Path(evaluating_pipeline_params.feature_encoded_classifier_path)
    preprocessing_pipeline=joblib.load(preprocessor_file_path)
    label_encoder_file_path=BASE_DIR/Path(evaluating_pipeline_params.label_encoder_classifier_path)
    label_encoder=joblib.load(label_encoder_file_path)
    feature_store_path=BASE_DIR/Path(evaluating_pipeline_params.feature_store_path)
    selected_feature_list=load_features_from_json(feature_store_path)
    model_type=evaluating_pipeline_params.model_type
    model_dir_path=BASE_DIR/evaluating_pipeline_params.model_dir_path
    experiment_dir=BASE_DIR/evaluating_pipeline_params.experiment_dir
    logger.info(f"laoding classifiers - preprocessing,label encoding and important features completed")
    logger.info(f"Validation of model {model_type} started")
    validate_model(preprocessing_pipeline,label_encoder,selected_feature_list,model_type, df_validation,model_dir_path,experiment_dir)
    logger.info(f"Validation of model {model_type} completed and validation metric is stored in {experiment_dir} ")

    return


@hydra.main(config_path="../configs", config_name="predict_config",version_base="1.1")
def predict_pipeline_start(cfg: DictConfig):
    os.chdir(hydra.utils.to_absolute_path(".."))
    schema = PredictingPipelineParamsSchema()
    params = schema.load(cfg)
    predict_pipeline(params)


if __name__ == "__main__":
    predict_pipeline_start()
