from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import joblib
import numpy as np

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
#   ML MODEL LOADING
# -----------------------------
model = joblib.load("exercise_model.pkl")
label_encoder = joblib.load("label_encoder.pkl")

class AngleData(BaseModel):
    angles: List[float]  # ← 10 angles from frontend

# -----------------------------
#   ML PREDICTION ENDPOINT
# -----------------------------
@app.post("/predict")
def predict_exercise(data: AngleData):
    arr = np.array(data.angles).reshape(1, -1)
    pred = model.predict(arr)[0]
    exercise_label = label_encoder.inverse_transform([pred])[0]
    return {"predicted_exercise": exercise_label}


# -----------------------------
#   YOUR ORIGINAL CODE
# -----------------------------
class PostureData(BaseModel):
    user_id: str
    exercise: str
    posture_score: float
    reps: int

data_store: List[PostureData] = []

@app.post("/posture")
def save_posture(data: PostureData):
    data_store.append(data)
    return {"message": "Data saved", "count": len(data_store)}

@app.get("/posture")
def get_posture():
    return data_store
