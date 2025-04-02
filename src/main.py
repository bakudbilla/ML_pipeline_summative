
import os
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel
from sklearn.calibration import LabelEncoder
import tensorflow as tf
import numpy as np
import pandas as pd
import database
import preprocessing
from preprocessing import preprocess_data
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from database import collection
from model import train_model, load_existing_model
from model import train_model  
from fastapi.middleware.cors import CORSMiddleware


# Load trained model
MODEL_PATH = os.path.abspath("../models/obesity_model.h5")
model = tf.keras.models.load_model(MODEL_PATH)


CLASS_MAPPING = {
    0: "Underweight",
    1: "Normal weight",
    2: "Overweight",
    3: "Obese"
}

class ObesityInput(BaseModel):
    Age: float
    Gender: str
    Height: float
    Weight: float
    BMI: float
    PhysicalActivityLevel: float

app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"],  
)
@app.get("/")
def welcome():
    return {"message": "Welcome to Obesi-Check"}



@app.post("/predict")
def predict_obesity(data: ObesityInput):
    df = pd.DataFrame([data.dict()])
    X_processed = preprocess_data(df, training=False)
    
    if len(X_processed.shape) == 1:
        X_processed = np.expand_dims(X_processed, axis=0)

    prediction = model.predict(X_processed)
    predicted_index = int(np.argmax(prediction, axis=1)[0])
    predicted_label = CLASS_MAPPING.get(predicted_index, "Unknown")

    return {"predicted_obesity_category": predicted_label}


@app.post("/upload/")
def upload_file(file: UploadFile = File(...)):
    database.clear_all_data()

    df = pd.read_csv(file.file)

    if df.empty:
        return {"error": "Uploaded file is empty"}

    
    required_columns = ["Age", "Gender", "Height", "Weight", "BMI", "PhysicalActivityLevel", "ObesityCategory"]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        return {"error": f"Missing required columns: {missing_columns}"}

    # Convert data types
    df = df.astype({
        "Age": float,
        "Gender": str,
        "Height": float,
        "Weight": float,
        "BMI": float,
        "PhysicalActivityLevel": float,
        "ObesityCategory": str
    })

    # Remove duplicates
    previous_count = len(df)
    df.drop_duplicates(inplace=True)
    final_count = len(df)

    # Insert into MongoDB
    inserted_data = database.insert_data(df.to_dict(orient="records"))

    return {
        "message": "Data loaded  to mongoDB",
        "rows_received": previous_count,
        "rows_inserted": final_count
    }


@app.post("/retrain")
def retrain_model():
    #fetching data from the mongoDB database
    data = database.fetch_all_data()

    print(f"Retrieved {len(data)} records from MongoDB (Type: {type(data)})")

    #saving data to a dataframe
    df = pd.DataFrame(data)
    
    X_train, y_train = preprocessing.preprocess_data(df, training=True)

    #retrain Model
    result = train_model(df)

    return {
        "message": "Model retrained successfully",
        "final_val_loss": result["final_val_loss"],
        "final_val_accuracy": result["final_val_accuracy"],
        "final_precision": result["final_precision"],
        "final_recall": result["final_recall"],
        "final_f1_score": result["final_f1_score"]
    }
