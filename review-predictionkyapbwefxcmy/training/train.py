import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

import mlflow
import os
import sys

import mlflow.xgboost

# adds the parent directory to path in case you have sibling folders with stuff you need
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import utils.utils

# List of required environment variables
REQUIRED_ENV_VARS = ["MLFLOW_TRACKING_URI", "AZURE_STORAGE_CONNECTION_STRING"]

# Check if all required variables are set
missing_vars = [var for var in REQUIRED_ENV_VARS if var not in os.environ]
if missing_vars:
    print(f"Error: Missing required environment variables: {', '.join(missing_vars)}")
    sys.exit(1)

# If all required variables are set, proceed
print("All required environment variables are set.")

mlflow.set_tracking_uri(os.environ['MLFLOW_TRACKING_URI'])

def main():
    # Prepare training data
    df = pd.read_csv('data/Digital_Music.csv')

    processed_df = utils.utils.preprocess_data(df)

    processed_df['combined_text'] = processed_df['title'] + " " + processed_df['text']
    y = processed_df['positive_review']

    tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X = tfidf.fit_transform(processed_df['combined_text'])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    smote = SMOTE(random_state=42)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

    with mlflow.start_run():
        mlflow.log_params({
            'objective': 'multi:softmax',
            'num_class': len(set(y)),
            'eval_metric': 'mlogloss',
            'use_label_encoder': False
        })
        
        xgb = XGBClassifier(
            objective='multi:softmax',
            num_class=len(set(y)), 
            eval_metric='mlogloss',
            use_label_encoder=False
        )

        xgb.fit(X_train_balanced, y_train_balanced)
        
        y_pred = xgb.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')

        mlflow.log_metrics({
            'accuracy': accuracy,
            'f1_score': f1,
            'precision': precision,
            'recall': recall
        })
        
        mlflow.xgboost.log_model(xgb, 'model')
        
        tfidf_path = "tfidf_vectorizer.pkl"
        joblib.dump(tfidf, tfidf_path)

        # Log the TfidfVectorizer as an artifact
        mlflow.log_artifact(tfidf_path)

if __name__ == "__main__":
    main()