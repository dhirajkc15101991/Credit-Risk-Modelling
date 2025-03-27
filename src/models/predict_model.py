from sklearn.metrics import accuracy_score,confusion_matrix,precision_score,recall_score,f1_score,classification_report,r2_score
import json
import os
from pathlib import Path
import joblib

def validate_model(preprocessing_pipeline,label_encoder,selected_feature_list,model_type, df_validation,model_dir_path,experiment_dir):
    """
    Loads the best saved model and evaluates it on the validation dataset.

    Parameters:
    - experiment_log: Path to the experiment JSON log.
    - X_val: Validation features.
    - y_val: Validation labels.

    Returns:
    - Validation accuracy.
    """
    #Selected only important features
    X_val=df_validation[selected_feature_list]
    y_val=df_validation['Approved_Flag']
    #Encode the features/input
    X_val_encoded=preprocessing_pipeline.transform(X_val)
    #Encode the label/taregt
    y_val_encoded=label_encoder.transform(y_val)

    if model_type=="RandomForestClassifier":
        model_name="model_"+model_type+".pkl"
    model_path=model_dir_path/model_name
    #load the model
    best_model = joblib.load(model_path)
    # Evaluate on validation set
    y_val_pred = best_model.predict(X_val_encoded)
    val_acc = accuracy_score(y_val_encoded, y_val_pred)
    # store the experiments
     # Experiment log
    experiment_data = {
        "model_name": model_name,
        "Testing_accuracy": val_acc,
    }
      # Save experiment log
    exp_log_path = os.path.join(experiment_dir, f"{model_type}_testing_experiment.json")
    with open(exp_log_path, "w") as f:
        json.dump(experiment_data, f, indent=4)

    print(experiment_data)

    # # Load saved best model
    # model_path = experiment_data["model_path"]
    # best_model = joblib.load(model_path)

    return val_acc