
import pandas as pd
import joblib
import os
import numpy as np
from huggingface_hub import hf_hub_download

from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import (
    BaggingClassifier,
    RandomForestClassifier,
    AdaBoostClassifier,
    GradientBoostingClassifier,
)
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn import metrics

# --- Global constants for Hugging Face ---
REPO_ID = "Dattaluri/TourismPackagePrediction"
PROCESSED_DATA_DIR_HF = "processed_data"
TRAINED_MODELS_DIR = "trained_models"

def get_metrics_score(model, X_train, y_train, X_test, y_test, flag=True):
    """
    model : classifier to predict values of X

    """
    # defining an empty list to store train and test results
    score_list=[]

    # predicting on train and tests
    pred_train = model.predict(X_train)
    pred_test = model.predict(X_test)

    # accuracy of the model
    train_acc = model.score(X_train,y_train)
    test_acc = model.score(X_test,y_test)

    # recall of the model
    train_recall = metrics.recall_score(y_train,pred_train)
    test_recall = metrics.recall_score(y_test,pred_test)

    # precision of the model
    train_precision = metrics.precision_score(y_train,pred_train)
    test_precision = metrics.precision_score(y_test,pred_test)

    # f1_score of the model
    train_f1 = metrics.f1_score(y_train,pred_train)
    test_f1 = metrics.f1_score(y_test,pred_test)

    # populate the score_list
    score_list.extend((train_acc,test_acc,train_recall,test_recall,train_precision,test_precision,train_f1,test_f1))

    # If the flag is set to True then only the following print statements will be dispayed. The default value is set to True.
    if flag == True:
        print("Accuracy on training set : ",train_acc)
        print("Accuracy on test set : ",test_acc)
        print("Recall on training set : ",train_recall)
        print("Recall on test set : ",test_recall)
        print("Precision on training set : ",train_precision)
        print("Precision on test set : ",test_precision)
        print("F1 on training set : ",train_f1)
        print("F1 on test set : ",test_f1)
    return score_list # returning the list with train and test scores



if __name__ == "__main__":
    # Create trained_models directory if it doesn't exist
    os.makedirs(TRAINED_MODELS_DIR, exist_ok=True)

    # --- Download and Load Data from Hugging Face ---
    print("\n--- Loading data from Hugging Face ---")
    try:
        # Download and load X_train
        X_train_path = hf_hub_download(repo_id=REPO_ID, filename=os.path.join(PROCESSED_DATA_DIR_HF, 'X_train.csv'))
        X_train = pd.read_csv(X_train_path)
        # Download and load X_test
        X_test_path = hf_hub_download(repo_id=REPO_ID, filename=os.path.join(PROCESSED_DATA_DIR_HF, 'X_test.csv'))
        X_test = pd.read_csv(X_test_path)
        # Download and load y_train
        y_train_path = hf_hub_download(repo_id=REPO_ID, filename=os.path.join(PROCESSED_DATA_DIR_HF, 'y_train.csv'))
        y_train = pd.read_csv(y_train_path).iloc[:, 0] # .iloc[:, 0] to get Series
        # Download and load y_test
        y_test_path = hf_hub_download(repo_id=REPO_ID, filename=os.path.join(PROCESSED_DATA_DIR_HF, 'y_test.csv'))
        y_test = pd.read_csv(y_test_path).iloc[:, 0] # .iloc[:, 0] to get Series
        # Download and load X_train_encoded_columns
        X_train_encoded_columns_path = hf_hub_download(repo_id=REPO_ID, filename=os.path.join(PROCESSED_DATA_DIR_HF, 'X_train_encoded_columns.joblib'))
        X_train_encoded_columns = joblib.load(X_train_encoded_columns_path)

        print("All data and column names loaded successfully.")

    except Exception as e:
        print(f"Error loading data from Hugging Face: {e}")
        exit(1)

    # --- One-Hot Encode Categorical Features and Align Columns ---
    print("\n--- One-hot encoding categorical features ---")
    # Apply one-hot encoding to X_train and X_test
    X_train_encoded = pd.get_dummies(X_train, drop_first=True)
    X_test_encoded = pd.get_dummies(X_test, drop_first=True)

    # Reindex X_test_encoded to align with X_train_encoded_columns (from joblib)
    X_train_encoded = X_train_encoded.reindex(columns=X_train_encoded_columns, fill_value=0)
    X_test_encoded = X_test_encoded.reindex(columns=X_train_encoded_columns, fill_value=0)
    print("One-hot encoding applied and columns aligned.")
    print(f"X_train_encoded shape: {X_train_encoded.shape}")
    print(f"X_test_encoded shape: {X_test_encoded.shape}")

    # --- Define Models and Hyperparameters ---
    print("\n--- Initializing models and hyperparameter grids ---")
    # Decision Tree Classifier
    dt_model = DecisionTreeClassifier(random_state=1)
    dt_param_grid = {
        'max_depth': np.arange(2, 10, 2),
        'min_samples_leaf': np.arange(1, 10, 2),
        'criterion': ['gini', 'entropy']
    }

    # Bagging Classifier
    bg_model = BaggingClassifier(random_state=1)
    bg_param_grid = {
        'n_estimators': [50, 100, 150],
        'max_features': [0.7, 0.8, 0.9],
        'max_samples': [0.7, 0.8, 0.9]
    }

    # Random Forest Classifier
    rf_model = RandomForestClassifier(random_state=1)
    rf_param_grid = {
        'n_estimators': [50, 100, 150],
        'max_depth': [5, 10, 15],
        'min_samples_leaf': [1, 2, 4]
    }

    # AdaBoost Classifier
    ada_model = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), random_state=1)
    ada_param_grid = {
        'n_estimators': [50, 100, 150],
        'learning_rate': [0.01, 0.1, 0.5]
    }

    # Gradient Boosting Classifier
    gb_model = GradientBoostingClassifier(random_state=1)
    gb_param_grid = {
        'n_estimators': [50, 100, 150],
        'learning_rate': [0.01, 0.1, 0.5],
        'max_depth': [3, 5, 7]
    }

    # XGBoost Classifier
    xgb_model = XGBClassifier(random_state=1, eval_metric='logloss')
    xgb_param_grid = {
        'n_estimators': [50, 100, 150],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7]
    }
    print("Models and parameter grids defined.")

    models = {
        'Decision Tree': (dt_model, dt_param_grid),
        'Bagging': (bg_model, bg_param_grid),
        'Random Forest': (rf_model, rf_param_grid),
        'AdaBoost': (ada_model, ada_param_grid),
        'Gradient Boosting': (gb_model, gb_param_grid),
        'XGBoost': (xgb_model, xgb_param_grid)
    }

    model_performance = {}
    best_f1_score = -1
    best_overall_model = None
    best_overall_model_name = ""

    # --- Train, Tune, and Evaluate Each Model ---
    for model_name, (model_estimator, param_grid) in models.items():
        print(f"\n--- Tuning {model_name} ---")
        grid_search = GridSearchCV(
            estimator=model_estimator,
            param_grid=param_grid,
            cv=5,
            scoring='f1',
            return_train_score=True,
            verbose=2
        )

        grid_search.fit(X_train_encoded, y_train)

        print(f"Best parameters for {model_name}: {grid_search.best_params_}")
        print(f"Best F1-score on cross-validation for {model_name}: {grid_search.best_score_:.4f}")

        best_model = grid_search.best_estimator_

        print(f"\n--- {model_name} (Best Estimator) Metrics ---")
        metrics_scores = get_metrics_score(best_model, X_train_encoded, y_train, X_test_encoded, y_test, flag=True)
        test_f1 = metrics_scores[7]
        model_performance[model_name] = test_f1

        if test_f1 > best_f1_score:
            best_f1_score = test_f1
            best_overall_model = best_model
            best_overall_model_name = model_name

    # --- Summary of Model Performance ---
    print("\n--- Summary of Model Performance (Test Set F1-scores) ---")
    performance_df = pd.DataFrame(model_performance.items(), columns=['Model', 'Test F1-score'])
    performance_df = performance_df.sort_values(by='Test F1-score', ascending=False).reset_index(drop=True)
    print(performance_df)

    print(f"\nRecommendation: The {best_overall_model_name} Classifier is the best performing model based on the F1-score of {best_f1_score:.4f} on the test set.")

    # --- Save the Best Model ---
    if best_overall_model:
        model_save_path = os.path.join(TRAINED_MODELS_DIR, f'{best_overall_model_name.lower().replace(" ", "_")}_model.joblib')
        joblib.dump(best_overall_model, model_save_path)
        print(f"Best model ({best_overall_model_name}) saved to '{model_save_path}'.")
    else:
        print("No best model found to save.")
