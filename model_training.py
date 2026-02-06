import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    BaggingClassifier,
    RandomForestClassifier,
    AdaBoostClassifier,
    GradientBoostingClassifier
)
from xgboost import XGBClassifier
from sklearn import metrics
from huggingface_hub import hf_hub_download

# --- Metrics Score Function (Modified to accept X_train, X_test, y_train, y_test) ---
def get_metrics_score(model, X_train, X_test, y_train, y_test, flag=True):
    """
    model : classifier to predict values of X
    X_train, X_test, y_train, y_test: training and testing data
    """
    score_list = []

    pred_train = model.predict(X_train)
    pred_test = model.predict(X_test)

    train_acc = model.score(X_train, y_train)
    test_acc = model.score(X_test, y_test)

    train_recall = metrics.recall_score(y_train, pred_train)
    test_recall = metrics.recall_score(y_test, pred_test)

    train_precision = metrics.precision_score(y_train, pred_train)
    test_precision = metrics.precision_score(y_test, pred_test)

    train_f1 = metrics.f1_score(y_train, pred_train)
    test_f1 = metrics.f1_score(y_test, pred_test)

    score_list.extend(
        (
            train_acc,
            test_acc,
            train_recall,
            test_recall,
            train_precision,
            test_precision,
            train_f1,
            test_f1,
        )
    )

    if flag:
        print(f"Accuracy on training set : {train_acc}")
        print(f"Accuracy on test set : {test_acc}")
        print(f"Recall on training set : {train_recall}")
        print(f"Recall on test set : {test_recall}")
        print(f"Precision on training set : {train_precision}")
        print(f"Precision on test set : {test_precision}")
        print(f"F1 on training set : {train_f1}")
        print(f"F1 on test set : {test_f1}")
    return score_list

def run_model_training():
    # --- Define paths and load data ---
    repo_id = "Dattaluri/TourismPackagePrediction"
    processed_data_dir = "processed_data"
    trained_models_dir = "trained_models"

    os.makedirs(trained_models_dir, exist_ok=True)

    print("Downloading processed data from Hugging Face...")
    try:
        X_train_path = hf_hub_download(repo_id=repo_id, filename=os.path.join(processed_data_dir, 'X_train.csv'))
        X_test_path = hf_hub_download(repo_id=repo_id, filename=os.path.join(processed_data_dir, 'X_test.csv'))
        y_train_path = hf_hub_download(repo_id=repo_id, filename=os.path.join(processed_data_dir, 'y_train.csv'))
        y_test_path = hf_hub_download(repo_id=repo_id, filename=os.path.join(processed_data_dir, 'y_test.csv'))
        cols_path = hf_hub_download(repo_id=repo_id, filename=os.path.join(processed_data_dir, 'X_train_encoded_columns.joblib'))

        X_train = pd.read_csv(X_train_path)
        X_test = pd.read_csv(X_test_path)
        y_train = pd.read_csv(y_train_path).iloc[:, 0]
        y_test = pd.read_csv(y_test_path).iloc[:, 0]
        X_train_encoded_columns = joblib.load(cols_path)

        print("Processed data and column names loaded successfully.")

    except Exception as e:
        print(f"Error loading processed data from Hugging Face: {e}")
        return

    # --- One-hot encode categorical features and align columns ---
    print("Applying one-hot encoding and aligning columns...")
    X_train_encoded = pd.get_dummies(X_train, drop_first=True)
    X_test_encoded = pd.get_dummies(X_test, drop_first=True)

    # Reindex to ensure same order of columns, crucial for consistent model fitting
    # Add missing columns with fill_value=0 and align order
    X_train_encoded = X_train_encoded.reindex(columns=X_train_encoded_columns, fill_value=0)
    X_test_encoded = X_test_encoded.reindex(columns=X_train_encoded_columns, fill_value=0)
    print("One-hot encoding and column alignment complete.")

    # --- Initialize models and hyperparameter grids ---
    models = {
        "Decision Tree": {
            "estimator": DecisionTreeClassifier(random_state=1),
            "param_grid": {
                'max_depth': np.arange(2, 10, 2),
                'min_samples_leaf': np.arange(1, 10, 2),
                'criterion': ['gini', 'entropy']
            },
        },
        "Bagging": {
            "estimator": BaggingClassifier(random_state=1),
            "param_grid": {
                'n_estimators': [50, 100, 150],
                'max_features': [0.7, 0.8, 0.9],
                'max_samples': [0.7, 0.8, 0.9]
            },
        },
        "Random Forest": {
            "estimator": RandomForestClassifier(random_state=1),
            "param_grid": {
                'n_estimators': [50, 100, 150],
                'max_depth': [5, 10, 15],
                'min_samples_leaf': [1, 2, 4]
            },
        },
        "AdaBoost": {
            "estimator": AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), random_state=1),
            "param_grid": {
                'n_estimators': [50, 100, 150],
                'learning_rate': [0.01, 0.1, 0.5]
            },
        },
        "Gradient Boosting": {
            "estimator": GradientBoostingClassifier(random_state=1),
            "param_grid": {
                'n_estimators': [50, 100, 150],
                'learning_rate': [0.01, 0.1, 0.5],
                'max_depth': [3, 5, 7]
            },
        },
        "XGBoost": {
            "estimator": XGBClassifier(random_state=1, eval_metric='logloss'),
            "param_grid": {
                'n_estimators': [50, 100, 150],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7]
            },
        },
    }

    best_f1_score = -1
    best_model_name = ""
    best_model_overall = None

    model_performance = []

    # --- Train and evaluate models ---
    print("Starting model training and evaluation...")
    for name, model_info in models.items():
        print(f"\n--- Training {name} ---")
        grid_search = GridSearchCV(
            estimator=model_info["estimator"],
            param_grid=model_info["param_grid"],
            cv=5,
            scoring='f1',
            return_train_score=True,
            verbose=2,
        )
        grid_search.fit(X_train_encoded, y_train)

        print(f"Best parameters for {name}: {grid_search.best_params_}")
        print(f"Best F1-score on cross-validation for {name}: {grid_search.best_score_}")

        current_best_model = grid_search.best_estimator_
        metrics = get_metrics_score(current_best_model, X_train_encoded, X_test_encoded, y_train, y_test, flag=False)
        test_f1 = metrics[7]

        model_performance.append({
            'Model': name,
            'Test F1-score': test_f1
        })

        if test_f1 > best_f1_score:
            best_f1_score = test_f1
            best_model_name = name
            best_model_overall = current_best_model

    print("\n--- Model Performance Summary (Test Set F1-scores) ---")
    performance_df = pd.DataFrame(model_performance).sort_values(by='Test F1-score', ascending=False).reset_index(drop=True)
    print(performance_df)

    print(f"\nRecommendation: The {best_model_name} Classifier is the best performing model based on the F1-score of {best_f1_score:.4f} on the test set.")

    # --- Save the best model ---
    if best_model_overall:
        model_save_path = os.path.join(trained_models_dir, 'best_model.joblib')
        joblib.dump(best_model_overall, model_save_path)
        print(f"Best model ({best_model_name}) saved to '{model_save_path}'")
    else:
        print("No best model identified or saved.")

if __name__ == "__main__":
    run_model_training()
