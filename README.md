Credit risk modelling
=============================

Project Organization
------------
    ├── configs
    │   └── feature_params     <- Configs for features
    │   └── path_config        <- Configs for all needed paths
    │   └── splitting_params   <- Configs for splitting params
    │   └── train_params       <- Configs for logreg and randomforest models parametres
    │   └── predict_config.yaml   <- Config for prediction pipline
    │   └── train_config.yaml   <- Config for train pipline
    │ 
    ├── data
    │   └── raw            <- The original, immutable data dump.
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts for splitting dataset to train and test
    │   │   └── make_dataset.py
    │   │
    │   ├── entities       <- Scripts for creating dataclasses
    │   │    
    │   │
    │   ├── features              <- Scripts to turn raw data into features for modeling
    │   │   └── feature_encoding_with_classifier.py
    |   |   └── feature_selection.py,
                handling_null_value.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── train_model.py
    │   │   └── predict_model.py
    |   |
    │   ├── outputs       <- Hydra logs ---Currently we are doing console logging only.
    │   │   
    │   ├──  utils        <- Scripts for serialized models, reading data
    │   |    └── utils.py
    |   |
    |   ├── predict_pipeline.py   <- pipeline for making predictions
    |   |
    |   └── train_pipeline.py     <- pipeline for model training
    |
    ├── tests              <- tests for the project
    ├── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io
    ├── LICENSE
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    ├── README.md          <- The top-level README for developers using this project.


--------
Run training
------------
These configs should be filled out for running train:

    ├── configs
    │   └── feature_params
    │   │   └── features.yaml   <- Config for feature description
    │   │
    │   ├── path_config           
    │   │   └── path_config.yaml <- Config with all paths: data, models, etc
    │   │
    │   ├── splitting_params
    │   │   └── splitting_params.yaml <- Config with parametes for split
    │   │
    │   ├── train_params
    |   |   └── logreg.yaml          <- Config for a model logisticregression
    │   │   └── rf.yaml              <- Config for a model randomforest
    │   │
    │   ├── train_config.yaml      <- Config for train_pipline for Hydra

Run model train:  `python src/train_pipeline.py`

--------
Run prediction
--------
These configs should be filled out for running predict:

    ├── configs
        └── predict_config.yaml <- path to models and transformers.
        
Run predict:  `python src/predict_pipeline.py`
