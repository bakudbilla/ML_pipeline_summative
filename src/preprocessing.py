import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
import os
from imblearn.over_sampling import SMOTE
from collections import Counter

SCALER_PATH = os.path.abspath("../models/scaker.pkl")
ENCODER_PATH = os.path.abspath("../models/encoder.pkl")

# Initialize objects
scaler = joblib.load(SCALER_PATH) if os.path.exists(SCALER_PATH) else StandardScaler()
label_encoder = joblib.load(ENCODER_PATH) if os.path.exists(ENCODER_PATH) else LabelEncoder()

def preprocess_data(df, training=False):
    
    required_features = ["Age", "Gender", "Height", "Weight", "BMI", "PhysicalActivityLevel"]
    
    if training:
        y = df["ObesityCategory"]
        df = df.drop(columns=["ObesityCategory"])
    else:
        y = None

    if "Gender" in df.columns:
        df["Gender"] = df["Gender"].astype(str).str.lower().map({"female": 0, "male": 1})

    if training:
        y_encoded = label_encoder.fit_transform(y)
        joblib.dump(label_encoder, ENCODER_PATH)

        class_dist = Counter(y_encoded)
        if min(class_dist.values()) / max(class_dist.values()) < 0.5:
            X_res, y_res = SMOTE().fit_resample(df, y_encoded)
        else:
            X_res, y_res = df, y_encoded

        X_scaled = scaler.fit_transform(X_res)
        joblib.dump(scaler, SCALER_PATH)
        
        return X_scaled, y_res
    else:
        X_scaled = scaler.transform(df[required_features])
        return X_scaled
