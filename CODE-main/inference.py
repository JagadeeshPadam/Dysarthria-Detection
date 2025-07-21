import pickle
import numpy as np
from preprocess.torgo_preprocess import preprocess_torgo
from preprocess.uaspeech_preprocess import preprocess_uaspeech

# Load models
model_torgo = None
model_uaspeech = None

try:
    with open("models/Torgo_Models/AdaBoost_model.pkl", "rb") as f:
        model_torgo = pickle.load(f)
        print("Torgo model loaded successfully!")
except Exception as e:
    print("Error loading Torgo model:", e)

try:
    with open("models/UASpeech_Models/Ensemble_model.pkl", "rb") as f:
        model_uaspeech = pickle.load(f)
        print("UASpeech model loaded successfully!")
except Exception as e:
    print("Error loading UASpeech model:", e)

def predict(audio_path, dataset):
    """Predict function that processes audio and makes predictions based on the selected dataset."""
    
    if dataset == "torgo":
        if model_torgo is None:
            return "Error: Torgo model is not loaded."
        features = preprocess_torgo(audio_path)
        prediction = model_torgo.predict([features])
    
    elif dataset == "uaspeech":
        if model_uaspeech is None:
            return "Error: UASpeech model is not loaded."
        features = preprocess_uaspeech(audio_path)
        prediction = model_uaspeech.predict([features])
    
    else:
        return "Error: Invalid dataset name. Choose either 'torgo' or 'uaspeech'."

    return str(prediction[0])

# import os
# import time
# def predict(audio_path):
#     """Check if the filename starts with 'dys' and return classification."""
#     time.sleep(3) 
#     filename = os.path.basename(audio_path)  # Extract filename from path
#     if filename.lower().startswith("dys"):
#         return "Dysarthric"
#     else:
#         return "Controlled"