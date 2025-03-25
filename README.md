StumbleUpon Evergreen Classification Challenge
==============================
This project based on https://www.kaggle.com/c/stumbleupon    
"StumbleUpon is a user-curated web content discovery engine that recommends relevant, high quality pages and media to its users, based on their interests. While some pages we recommend, such as news articles or seasonal recipes, are only relevant for a short period of time, others maintain a timeless quality and can be recommended to users long after they are discovered."
   
A short description of the project.
------------
The primary objective of this project is to establish a well-organized project structure and effectively implement diverse techniques.   
Project structure is based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>.   
For config handling  <a target="_blank" href="https://hydra.cc/docs/intro/">Hydra library</a>  is used .   
Notebooks folder contains links to colab notebooks with shap value and DoWhy explorations (ToDo: clean and translate notebooks).   

To be done:
1. Translate notebooks.
2. Improve parameters optimization

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
    │   │   └── build_features.py
    |   |   └── custom_scaler.py  <- Custom scaler transformer
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    |   |
    │   ├── outputs       <- Hydra logs
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
