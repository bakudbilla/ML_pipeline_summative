import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping, Callback, ReduceLROnPlateau
from tensorflow.keras import regularizers
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score  # <-- Add this import
import preprocessing

MODEL_PATH = os.path.abspath("../models/obesity_model.h5")

class MetricsCallback(Callback):
    def __init__(self, validation_data=None):
        super().__init__()
        self.validation_data = validation_data
        self.final_precision = None
        self.final_recall = None
        self.final_f1_score = None
        self.final_val_loss = None
        self.final_val_accuracy = None

    def on_train_end(self, logs=None):
        if self.validation_data:
            X_val, y_val = self.validation_data
            y_pred = self.model.predict(X_val)

            if len(y_pred.shape) > 1 and y_pred.shape[1] == 1:
                y_pred_classes = (y_pred > 0.5).astype(int).flatten()
            elif len(y_pred.shape) > 1:
                y_pred_classes = tf.argmax(y_pred, axis=-1).numpy()
            else:
                y_pred_classes = (y_pred > 0.5).astype(int)

        
            self.final_precision = precision_score(y_val, y_pred_classes, average='macro', zero_division=0)
            self.final_recall = recall_score(y_val, y_pred_classes, average='macro', zero_division=0)
            self.final_f1_score = f1_score(y_val, y_pred_classes, average='macro', zero_division=0)
            
            self.final_val_loss = logs.get('val_loss')
            self.final_val_accuracy = logs.get('val_accuracy')

            print(f"Final Metrics: Precision: {self.final_precision:.4f}, Recall: {self.final_recall:.4f}, F1 Score: {self.final_f1_score:.4f}")
            print(f"Final Validation Loss: {self.final_val_loss}")
            print(f"Final Validation Accuracy: {self.final_val_accuracy}")

def load_existing_model():
    if os.path.exists(MODEL_PATH):
        model = load_model(MODEL_PATH)
        print(f"Model loaded. Expected input shape: {model.input_shape}")
        return model
    print("No existing model found. Train a new one first.")
    return None

def train_model(df):
    if df.empty:
        raise ValueError("No training data available.")
    
    print("Checking data format...")
    print(f"Data type: {type(df)}")
    print(f"Data shape: {df.shape}")
    
    print("Data sample before preprocessing:")
    print(df.head(2))
    
    X, y = preprocessing.preprocess_data(df, training=True)
    print("Preprocessing complete.")
    print(f"X shape: {X.shape}, y shape: {y.shape}")
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y if len(set(y)) > 1 else None)
    print(f"X_train shape: {X_train.shape}, X_val shape: {X_val.shape}")
    
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(24, activation='relu', input_shape=(X_train.shape[1],),
                             kernel_regularizer=regularizers.l2(0.03)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.5),
        
        tf.keras.layers.Dense(12, activation='relu', 
                             kernel_regularizer=regularizers.l2(0.03)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.5),
        
        tf.keras.layers.Dense(len(set(y_train)), activation='softmax')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy', 
        metrics=['accuracy']
    )

    metrics_callback = MetricsCallback(validation_data=(X_val, y_val))
    
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=0.0001
    )

    history = model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=16,
        validation_data=(X_val, y_val),
        callbacks=[early_stopping, reduce_lr, metrics_callback],
        verbose=1
    )

#final loss and accuracy scores
    val_loss = metrics_callback.final_val_loss
    val_accuracy = metrics_callback.final_val_accuracy

    # If metrics are still None, log a warning
    if val_loss is None or val_accuracy is None:
        print("Warning: Final validation loss or accuracy is None.")
    
    model.save(MODEL_PATH)
    print("Model retrained and saved successfully.")
    
    return {
        "status": "success",
        "final_val_loss": val_loss,
        "final_val_accuracy": val_accuracy,
        "final_precision": metrics_callback.final_precision,
        "final_recall": metrics_callback.final_recall,
        "final_f1_score": metrics_callback.final_f1_score
    }
