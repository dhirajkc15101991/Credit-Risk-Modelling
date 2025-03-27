
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score,confusion_matrix,precision_score,recall_score,f1_score,classification_report,r2_score
import joblib
import os
import json

def training_model(param_distributions,preprocessing_pipeline,label_encoder,X_train,y_train,save_dir_path,model_name,experiment_dir):
    #RandomForestClassifier
    #Encode the features/input
    if model_name=='RandomForestClassifier':
        model=RandomForestClassifier()

    X_train_encoded=preprocessing_pipeline.transform(X_train)
    #Encode the label/taregt
    y_train_encoded=label_encoder.transform(y_train)
    #fit the model
    # Hyperparameter tuning using RandomizedSearchCV
    search = RandomizedSearchCV(model, param_distributions=param_distributions, n_iter=4, cv=3, n_jobs=-1, random_state=42)
    search.fit(X_train_encoded, y_train_encoded)

    # Best model & hyperparameters
    best_model = search.best_estimator_
    best_params = search.best_params_

    # Save model
    path_to_save=os.path.join(save_dir_path, f"model_{model_name}.pkl")
    joblib.dump(best_model, path_to_save)

    # Training metric (accuracy)
    y_train_pred = best_model.predict(X_train_encoded)
    train_acc = accuracy_score(y_train_encoded, y_train_pred)

    #log the experiment

     # Experiment log
    experiment_data = {
        "model_name": model_name,
        "best_params": best_params,
        "training_accuracy": train_acc,
        "model_path": path_to_save
    }

     # Save experiment log
    exp_log_path = os.path.join(experiment_dir, f"{model_name}_training_experiment.json")
    with open(exp_log_path, "w") as f:
        json.dump(experiment_data, f, indent=4)

    return experiment_data


