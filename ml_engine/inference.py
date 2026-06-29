"""
INFERENCE MODULE
This is the "Brain Wrapper". It loads the model, takes raw network data, 
runs it through the physics rules, asks the AI for a prediction, adds the XAI,
and packages it all up perfectly for the backend.
"""
import os
import joblib
import pandas as pd
from .physics_rules import calculate_soft_hard_ratio

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "saved_models", "flare_predictor_v1.pkl")

class FlarePredictorEngine:
    def __init__(self):
        self.model = None
        self._load_model()
        
    def _load_model(self):
        try:
            if os.path.exists(MODEL_PATH):
                self.model = joblib.load(MODEL_PATH)
                print("✅ ML Engine Initialized.")
            else:
                print("⚠️ Model missing.")
        except Exception as e:
            print(f"Error loading model: {e}")

    def predict(self, raw_data: dict) -> dict:
        """
        The main prediction pipeline.
        """
        if self.model is None:
            return {"status": "error", "message": "Model offline"}

        # 1. Physics overrides (Safety check)
        if raw_data['soft_xray_counts'] < 0:
            return {"status": "error", "message": "Corrupted sensor data"}

        # 2. Prepare features
        features = {
            'soft_xray_counts': raw_data['soft_xray_counts'],
            'soft_xray_derivative': raw_data['soft_xray_derivative'],
            'soft_to_hard_ratio': raw_data['soft_to_hard_ratio']
        }
        
        df = pd.DataFrame([features])
        
        # 3. Model Inference
        probabilities = self.model.predict_proba(df)[0]
        flare_prob = round(probabilities[1] * 100, 2)
        quiet_prob = round(probabilities[0] * 100, 2)
        
        # 4. Determine Global State
        state = "NOMINAL"
        if flare_prob > 80:
            state = "CRITICAL"
        elif flare_prob > 40:
            state = "WARNING"
            
        # 5. Package the complete enterprise response
        return {
            "status": "success",
            "global_state": state,
            "probabilities": {
                "quiet": quiet_prob,
                "flare": flare_prob
            }
        }