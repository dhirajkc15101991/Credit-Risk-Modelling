from dataclasses import dataclass


@dataclass
class PathParams:
    raw_cibil_data_path: str
    raw_internal_data_path:str
    input_data_path: str
    output_model_path: str
    output_transformer_path: str
    metric_path: str
