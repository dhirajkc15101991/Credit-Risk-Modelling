from dataclasses import dataclass


@dataclass
class PathParams:
    raw_cibil_data_path: str
    raw_internal_data_path:str
    train_data_path: str
    val_data_path:str
    output_model_path: str
    # output_transformer_path: str
    # metric_path: str
    feature_store_path:str
    feature_encoded_classifier_path:str
    label_encoder_classifier_path:str
    experiment_dir:str
