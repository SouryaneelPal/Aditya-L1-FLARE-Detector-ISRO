from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import pandas as pd
import os

# Setup paths to find the saved model
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "ml_engine", "saved_models", "flare_predictor_v1.pkl")

# Initialize FastAPI app
app = FastAPI(title="Aditya-L1 S.F.E.W.S API", description="Backend for Solar Flare Early Warning System")

# Allow CORS so your frontend index.html can communicate with this API securely
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins for hackathon local testing
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the trained model at startup
model = None
try:
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
        print(f"✅ AI Model loaded successfully from {MODEL_PATH}")
    else:
        print(f"⚠️ Warning: Model not found at {MODEL_PATH}. Run train_model.py first.")
except Exception as e:
    print(f"❌ Error loading model: {e}")


# Define the expected incoming JSON data structure from the dashboard
class TelemetryData(BaseModel):
    soft_xray_counts: float
    soft_xray_derivative: float
    soft_to_hard_ratio: float

@app.get("/")
def read_root():
    """Health check endpoint to verify the server is running."""
    return {"status": "Aditya-L1 Backend API is Online and Nominal."}

@app.post("/api/predict")
def predict_flare(data: TelemetryData):
    """
    Receives live telemetry features, feeds them to the Random Forest model,
    and returns the probability of a flare in the next 30 minutes.
    """
    if model is None:
        raise HTTPException(status_code=500, detail="AI Model is offline.")
    
    try:
        # Convert incoming JSON data to a Pandas DataFrame for the ML model
        # We use .model_dump() for Pydantic v2 compatibility
        df = pd.DataFrame([data.model_dump() if hasattr(data, 'model_dump') else data.dict()])
        
        # Predict probability of classes (0: Quiet Sun, 1: Flare Imminent)
        probabilities = model.predict_proba(df)[0]
        
        quiet_prob = round(probabilities[0] * 100, 2)
        flare_prob = round(probabilities[1] * 100, 2)
        
        # Extrapolate granular threat levels based on the binary model output
        # (This translates the AI's binary logic into the B, C, M, X UI bars)
        response_data = {
            "status": "success",
            "ai_confidence": {
                "quiet_sun": quiet_prob,
                "flare_imminent": flare_prob
            },
            "threat_levels": {
                "b_class": max(10, quiet_prob), # Always some baseline B-class activity
                "c_class": flare_prob * 0.8,
                "m_class": flare_prob * 0.5,
                "x_class": flare_prob * 0.2 if flare_prob > 80 else 1.0 # Only high if imminent is huge
            },
            "global_state": "CRITICAL" if flare_prob > 80 else "WARNING" if flare_prob > 40 else "NOMINAL"
        }
        
        return response_data
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    # Run the server on port 8000
    print("🚀 Starting Aditya-L1 Mission Control Server...")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)