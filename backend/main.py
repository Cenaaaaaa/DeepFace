from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from deepface import DeepFace
import numpy as np
import cv2
import base64
import pickle
import os
from datetime import datetime
import pandas as pd

app = FastAPI()

# CORS untuk React
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DB_FILE = "face_db.pkl"
LOG_FILE = "attendance_log.csv"

MODELS = {
    "Facenet": "Facenet",
    "Facenet512": "Facenet512"
}

THRESHOLDS = {
    "Facenet": 0.40,
    "Facenet512": 0.30
}

# Load DB
if os.path.exists(DB_FILE):
    with open(DB_FILE, "rb") as f:
        face_db = pickle.load(f)
else:
    face_db = {}

# Setup log
if not os.path.exists(LOG_FILE):
    pd.DataFrame(columns=["timestamp", "name", "nim", "model", "distance", "confidence"]).to_csv(LOG_FILE, index=False)

def base64_to_image(base64_str):
    header, encoded = base64_str.split(",", 1)
    img_bytes = base64.b64decode(encoded)
    img_array = np.frombuffer(img_bytes, dtype=np.uint8)
    return cv2.imdecode(img_array, cv2.IMREAD_COLOR)

def get_embedding(img, model_name):
    embedding = DeepFace.represent(
        img,
        model_name=model_name,
        enforce_detection=False
    )[0]["embedding"]
    return np.array(embedding)

class RegisterBody(BaseModel):
    nim: str
    name: str
    image: str

@app.post("/api/register")
def register(body: RegisterBody):
    img = base64_to_image(body.image)
    model_name = "Facenet512"
    emb = get_embedding(img, model_name)

    face_db[body.nim] = {
        "name": body.name,
        "embedding": emb
    }

    with open(DB_FILE, "wb") as f:
        pickle.dump(face_db, f)

    return {"success": True}

class AttendanceBody(BaseModel):
    image: str
    model: str

@app.post("/api/attendance")
def attendance(body: AttendanceBody):
    img = base64_to_image(body.image)
    model_name = MODELS[body.model]
    threshold = THRESHOLDS[body.model]

    emb = get_embedding(img, model_name)

    best_nim = None
    best_name = None
    best_distance = 999

    for nim, data in face_db.items():
        db_emb = data["embedding"]
        dist = np.linalg.norm(emb - db_emb)
        if dist < best_distance:
            best_distance = dist
            best_nim = nim
            best_name = data["name"]

    if best_distance > threshold:
        return {"success": False, "message": "Wajah tidak dikenali"}

    confidence = 1 - (best_distance / threshold)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    df = pd.read_csv(LOG_FILE)
    df.loc[len(df)] = [
        timestamp,
        best_name,
        best_nim,
        model_name,
        round(best_distance, 4),
        round(confidence, 4),
    ]
    df.to_csv(LOG_FILE, index=False)

    return {
        "success": True,
        "name": best_name,
        "nim": best_nim,
        "distance": float(best_distance),
        "confidence": float(confidence),
        "timestamp": timestamp
    }

@app.post("/api/upload-dataset")
async def upload_dataset(nim: str = Form(...), images: list[UploadFile] = File(...)):
    if nim not in face_db:
        return {"success": False, "message": "NIM belum terdaftar"}

    embeddings = []

    for img_file in images:
        img_bytes = await img_file.read()
        img_array = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        emb = get_embedding(img, "Facenet512")
        embeddings.append(emb)

    original_emb = face_db[nim]["embedding"]
    embeddings.append(original_emb)

    avg_emb = np.mean(embeddings, axis=0)

    face_db[nim]["embedding"] = avg_emb

    with open(DB_FILE, "wb") as f:
        pickle.dump(face_db, f)

    return {"success": True}

@app.get("/api/dashboard")
def dashboard():
    total_registered = len(face_db)
    df = pd.read_csv(LOG_FILE)
    total_attendance = len(df)

    today = datetime.now().strftime("%Y-%m-%d")
    today_attendance = len(df[df["timestamp"].str.contains(today)])

    logs = df.tail(20).to_dict(orient="records")

    return {
        "total_registered": total_registered,
        "total_attendance": total_attendance,
        "today_attendance": today_attendance,
        "logs": logs
    }
