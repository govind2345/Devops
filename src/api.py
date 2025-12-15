from fastapi import FastAPI
import joblib
import pandas as pd

app = FastAPI()

MODEL_PATH = "models/model_latest.joblib"

try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    model = None

@app.get("/")
def home():
    return {"status": "API Running"}

@app.post("/predict")
def predict(data: dict):
    df = pd.DataFrame([data])
    prediction = model.predict(df)[0]
    label = "anomaly" if prediction == -1 else "normal"
    return {"prediction": label}
