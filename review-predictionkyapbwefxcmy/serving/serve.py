from fastapi import FastAPI
import joblib
from pydantic import BaseModel
import uvicorn
import mlflow
import mlflow.xgboost
import os
import sys

# adds the parent directory to path in case you have sibling folders with stuff you need
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# List of required environment variables
REQUIRED_ENV_VARS = ["MLFLOW_RUN_ID", "AZURE_STORAGE_CONNECTION_STRING", "MLFLOW_TRACKING_URI"]

# Check if all required variables are set
missing_vars = [var for var in REQUIRED_ENV_VARS if var not in os.environ]
if missing_vars:
    print(f"Error: Missing required environment variables: {', '.join(missing_vars)}")
    sys.exit(1)

# If all required variables are set, proceed
print("All required environment variables are set.")

app = FastAPI()

model = mlflow.xgboost.load_model(f"runs:/{os.environ['MLFLOW_RUN_ID']}/model")

tfidf_path = f"runs:/{os.environ['MLFLOW_RUN_ID']}/tfidf_vectorizer.pkl"  # Adjust path to where you stored it

local_tfidf_path = mlflow.artifacts.download_artifacts(tfidf_path)

tfidf = joblib.load(local_tfidf_path)  # Load the TfidfVectorizer

class PredictRequest(BaseModel):
    review: str

@app.post('/predict')
def predict(request: PredictRequest):
    print(request.review)
    
    X_new = tfidf.transform([request.review])
    
    # Predict using the loaded XGBoost model
    y_pred = model.predict(X_new)
    
    
    return {"is_positive": int(y_pred[0]) == 1}
    
@app.get("/")
def root():
    return {"message": "Hello world"}

def main():
    uvicorn.run(app, host='0.0.0.0', port=int(os.environ['LISTEN_PORT']))
    
if __name__ == "__main__":
    main()