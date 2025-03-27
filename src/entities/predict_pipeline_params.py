from dataclasses import dataclass

from marshmallow_dataclass import class_schema


@dataclass()
class PredictingPipelineParams:

    validation_data_path: str
    feature_encoded_classifier_path: str
    label_encoder_classifier_path: str
    feature_store_path: str
    model_dir_path: str
    output_data_path: str
    model_type: str
    experiment_dir: str
    # input_data_path: str
    # output_data_path: str
    # pipeline_path: str
    # model_path: str
PredictingPipelineParamsSchema = class_schema(PredictingPipelineParams)
