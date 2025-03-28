from typing import Union

from dataclasses import dataclass
from .split_params import SplittingParams
from .feature_params import FeatureParams
from .optimizer_params import OptimizerParams
from .train_params import LogRegParams, RandomForestParams, MLPParams
from .path_params import PathParams
from marshmallow_dataclass import class_schema


@dataclass()
class TrainingPipelineParams:
    path_config: PathParams
    splitting_params: SplittingParams
    feature_params: FeatureParams
    train_params: RandomForestParams
    # train_params: Union[LogRegParams, RandomForestParams, MLPParams]


TrainingPipelineParamsSchema = class_schema(TrainingPipelineParams)
