from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import sys

# --- Enterprise Setup ---
# Add the project base directory to sys.path so we can import from ml_engine
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

# Import the Inference Engine Controller
from ml_engine.inference import FlarePredictorEngine

# --- App & Resource Initialization ---
app = FastAPI(
    title="Aditya-L1 S.F.E.W.S Enterprise API",
    description="Backend for Solar Flare Early Warning System"
)

# Allow CORS so your frontend (localhost:8080) can communicate with this API (localhost:8000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins for local development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the ML Engine Core once at startup
print("⚙️ Initializing Aditya-L1 AI Core...")
ml_engine = FlarePredictorEngine()
print("✅ AI Core Online and Ready.")

# --- Data Models ---
# Define the expected incoming JSON data structure from the dashboard
class TelemetryData(BaseModel):
    soft_xray_counts: float
    soft_xray_derivative: float
    soft_to_hard_ratio: float

# --- API Endpoints ---
@app.get("/")
def read_root():
    """Health check endpoint to verify the server is running."""
    return {"status": "Aditya-L1 Enterprise Backend API is Online."}

@app.post("/api/predict")
async def predict_flare(data: TelemetryData):
    """
    Main inference endpoint.
    Receives live telemetry features and passes them to the inference engine.
    """
    try:
        print(f"\n📡 Received Telemetry: {data}")
        
        # Safely convert the Pydantic object to a dictionary.
        data_dict = data.model_dump() if hasattr(data, 'model_dump') else data.dict()
        
        # Pass the dictionary to the engine's predict method
        response = ml_engine.predict(data_dict)
        
        if response.get("status") == "error":
            print(f"⚠️ Engine Error: {response.get('message')}")
            raise HTTPException(status_code=400, detail=response.get("message", "Unknown error"))
        
        # Reconstruct threat levels for the frontend UI progress bars
        flare_prob = response.get("probabilities", {}).get("flare", 0)
        quiet_prob = response.get("probabilities", {}).get("quiet", 100)
        
        response["threat_levels"] = {
            "b_class": max(10, quiet_prob), 
            "c_class": flare_prob * 0.8,
            "m_class": flare_prob * 0.5,
            "x_class": flare_prob * 0.2 if flare_prob > 80 else 1.0 
        }
        
        # Expose the 1-2 year macro forecast from the AI engine to the UI
        response["macro_forecast"] = response.get("macro_forecast", {})
        
        # Add a flag to prove to the frontend that this is real AI data
        response["source"] = "real_time_ai_inference"
        
        print(f"✅ Prediction Successful! State: {response.get('global_state')}")
        return response
        
    except Exception as e:
        print(f"❌ Prediction Error in API: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    # Run the server on port 8000
    print("🚀 Starting Mission Control Server...")
    # Because everything is in main.py, we run "main:app"
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)