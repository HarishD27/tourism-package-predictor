# for data manipulation
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
# for model training, tuning, and evaluation
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
# for model serialization
import joblib
# for creating a folder
import os
# for hugging face space authentication to upload files
from huggingface_hub import HfApi, create_repo, hf_hub_download
from huggingface_hub.utils import RepositoryNotFoundError
import mlflow

mlflow.set_tracking_uri("mlruns")
mlflow.set_experiment("MLOps_CICD_experiment")

api = HfApi(token=os.getenv("HF_TOKEN"))

repo_id = "Harish2710/tourism-package-prediction"

# Download files locally first
Xtrain_local = hf_hub_download(repo_id=repo_id, filename="data/Xtrain.csv", repo_type="dataset")
Xtest_local  = hf_hub_download(repo_id=repo_id, filename="data/Xtest.csv",  repo_type="dataset")
ytrain_local = hf_hub_download(repo_id=repo_id, filename="data/ytrain.csv", repo_type="dataset")
ytest_local  = hf_hub_download(repo_id=repo_id, filename="data/ytest.csv",  repo_type="dataset")

# Read from downloaded local paths
Xtrain = pd.read_csv(Xtrain_local)
Xtest  = pd.read_csv(Xtest_local)
ytrain = pd.read_csv(ytrain_local).squeeze()
ytest  = pd.read_csv(ytest_local).squeeze()

# Define numeric and categorical features
numeric_features = [
    'Age', 'CityTier', 'NumberOfPersonVisiting', 'PreferredPropertyStar',
    'NumberOfTrips', 'Passport', 'OwnCar', 'NumberOfChildrenVisiting',
    'MonthlyIncome', 'PitchSatisfactionScore', 'NumberOfFollowups', 'DurationOfPitch'
]

categorical_features = [
    'TypeofContact', 'Occupation', 'Gender', 'MaritalStatus', 'Designation', 'ProductPitched'
]

# Preprocessor
preprocessor = make_column_transformer(
    (StandardScaler(), numeric_features),
    (OneHotEncoder(handle_unknown='ignore'), categorical_features)
)

# Define base XGBoost Classifier
xgb_model = xgb.XGBClassifier(random_state=42, n_jobs=-1, eval_metric='logloss')

# Hyperparameter grid
param_grid = {
    'xgbclassifier__n_estimators': [50, 100, 150],
    'xgbclassifier__max_depth': [3, 5, 7],
    'xgbclassifier__learning_rate': [0.01, 0.05, 0.1],
    'xgbclassifier__subsample': [0.7, 0.8, 1.0],
    'xgbclassifier__colsample_bytree': [0.7, 0.8, 1.0],
    'xgbclassifier__reg_lambda': [0.1, 1, 10]
}

# Pipeline
model_pipeline = make_pipeline(preprocessor, xgb_model)

with mlflow.start_run():
    # Grid Search
    grid_search = GridSearchCV(model_pipeline, param_grid, cv=3, n_jobs=-1, scoring='accuracy')
    grid_search.fit(Xtrain, ytrain)

    # Log parameter sets
    results = grid_search.cv_results_
    for i in range(len(results['params'])):
        param_set = results['params'][i]
        mean_score = results['mean_test_score'][i]

        with mlflow.start_run(nested=True):
            mlflow.log_params(param_set)
            mlflow.log_metric("mean_accuracy", mean_score)

    # Best model
    mlflow.log_params(grid_search.best_params_)
    best_model = grid_search.best_estimator_

    # Predictions
    y_pred_train = best_model.predict(Xtrain)
    y_pred_test = best_model.predict(Xtest)

    # Metrics
    train_acc = accuracy_score(ytrain, y_pred_train)
    test_acc = accuracy_score(ytest, y_pred_test)

    train_f1 = f1_score(ytrain, y_pred_train)
    test_f1 = f1_score(ytest, y_pred_test)

    train_auc = roc_auc_score(ytrain, y_pred_train)
    test_auc = roc_auc_score(ytest, y_pred_test)

    # Log metrics
    mlflow.log_metrics({
        "train_Accuracy": train_acc,
        "test_Accuracy": test_acc,
        "train_F1": train_f1,
        "test_F1": test_f1,
        "train_AUC": train_auc,
        "test_AUC": test_auc
    })

    # Save the model locally
    model_path = "tourism_package/model_building/best_tourism_package_model_v1.joblib"
    joblib.dump(best_model, model_path)

    # Log the model artifact
    mlflow.log_artifact(model_path, artifact_path="model")
    print(f"Model saved as artifact at: {model_path}")

    # Upload to Hugging Face
    repo_id = "Harish2710/tourism-package-model"
    repo_type = "model"

    try:
        api.repo_info(repo_id=repo_id, repo_type=repo_type)
        print(f"Repo '{repo_id}' already exists. Using it.")
    except RepositoryNotFoundError:
        print(f"Repo '{repo_id}' not found. Creating new repo...")
        create_repo(repo_id=repo_id, repo_type=repo_type, private=False)
        print(f"Repo '{repo_id}' created.")

    files = ["best_tourism_package_model_v1.joblib"]

    for file_path in files:
      api.upload_file(
          path_or_fileobj=model_path,
          path_in_repo=f"model/{file_path}",
          repo_id="Harish2710/tourism-package-model",
          repo_type="model",
      )
