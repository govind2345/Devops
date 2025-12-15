from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class PredictRequest(BaseModel):
    bytes: int
    packets: int

@app.get("/")
def root():
    return {"status": "API Running"}

@app.post("/predict")
def predict(data: PredictRequest):
    try:
        anomaly_score = data.bytes / (data.packets + 1)
        is_anomaly = anomaly_score > 100

        return {
            "bytes": data.bytes,
            "packets": data.packets,
            "anomaly_score": anomaly_score,
            "anomaly": is_anomaly
        }
    except Exception as e:
        return {"error": str(e)}
