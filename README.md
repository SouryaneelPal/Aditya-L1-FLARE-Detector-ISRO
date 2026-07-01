☀️ Aditya-L1 Solar Flare Early Warning System (S.F.E.W.S)

Welcome to S.F.E.W.S! This project is a real-time, AI-driven mission control dashboard designed to predict catastrophic solar flares using data from ISRO's Aditya-L1 spacecraft.

When a massive X-Class solar flare hits Earth, it can cause global radio blackouts, cripple GPS systems, and damage power grids. By analyzing X-ray emissions from the sun, this system gives us the crucial warning time needed to brace for impact.

🚀 What does it do?

Ingests Space Data: It processes raw ISRO PRADAN data from two specific Aditya-L1 instruments:

SoLEXS: Measures soft X-rays (thermal plasma buildup).

HEL1OS: Measures hard X-rays (non-thermal explosions).

Thinks like an Astrophysicist: The AI is grounded in real solar physics ( specifically the Neupert Effect). It doesn't just look at numbers; it looks at the rate of change and the ratio between soft and hard X-rays.

Predicts the Future: A Random Forest Machine Learning model evaluates the telemetry and predicts the probability of a flare occurring within the next 30 minutes.

Visualizes the Threat: The frontend is a fully immersive "Mission Control" UI featuring a live 3D simulation of Earth's magnetic shield reacting to solar pressure, complete with audio sonification (Geiger counter effects).

🛠️ Tech Stack

Backend / ML Engine: Python, FastAPI, Scikit-Learn, Pandas, NumPy

Frontend: HTML5, Tailwind CSS, JavaScript (Vanilla)

Visualizations: Three.js (3D Magnetosphere), Observable Plot / D3.js (Telemetry charts)

Data Source: ISRO PRADAN (Aditya-L1 Mission Data)

🧠 How the AI Works (The Science)

Most of the time, the sun is quiet. To train our AI, we parsed thousands of minutes of baseline data and applied a statistical Noise Floor. We taught the AI that standard background radiation is normal.

However, we also anchored the model with Extreme Out-of-Distribution Events. When the AI sees a massive spike in the derivative of the soft X-ray count (meaning the sun is rapidly heating up) coupled with a sudden burst in hard X-rays, the Random Forest model flags an impending X-Class flare and triggers the system lockdown.

💻 How to Run It Locally

Everything is modular and designed to run smoothly on your local machine.

1. Prerequisites

Make sure you have Python 3.x installed. Then, install the required dependencies:

pip install fastapi uvicorn pandas scikit-learn joblib pydantic astropy


2. Train the Model (Optional but recommended)

We need to generate the "brain" of the system (flare_predictor_v1.pkl).
From the root of the project, run:

python ml_engine/train_model.py


Note: This will parse the dataset, apply the noise floor, inject our extreme event calibration anchor, and save the compiled model.

3. Start the Backend API

The FastAPI server acts as the bridge between the AI model and the dashboard.

python backend/main.py


The API will start running on http://localhost:8000.

4. Launch Mission Control (Frontend)

Because the frontend is pure HTML/JS/CSS, you don't need a web server to view it!
Just double-click the index.html file in your file explorer to open it in Chrome, Firefox, or Safari.

🎮 The "Demo" Mode (Inject Anomaly)

Because the sun is quiet 99% of the time, staring at a nominal dashboard during a presentation isn't very exciting.

We built an "Inject Anomaly" button into the dashboard. Clicking this safely simulates an apocalyptic thermal buildup.

The dashboard translates the visual graph data into raw ISRO sensor counts.

It applies a signal smoother (low-pass filter) to remove UI jitter.

It sends this clamped, simulated data to the Python backend.

The AI detects the massive Neupert Effect signature, latches the system into a CRITICAL state, triggers the audio alarms, and visually compresses the 3D Earth magnetosphere.

To return to normal operations, simply click Reset.

📂 Project Structure

├── backend/
│   ├── main.py             # FastAPI entry point
│   ├── routes.py           # API endpoints
│   └── config.py           # Environment configurations
├── data_pipeline/
│   ├── fetch_pradan.py     # ISRO data downloader & parser
│   └── preprocess.py       # Cleans and aligns SoLEXS/HEL1OS data
├── ml_engine/
│   ├── train_model.py      # Random forest training script
│   ├── inference.py        # Connects the model to the API
│   ├── physics_rules.py    # Astrophysics math (Neupert effect)
│   └── explainability.py   # XAI (Explainable AI) logic
├── index.html              # The Mission Control Dashboard
└── requirements.txt        # Python dependencies


Built with ❤️ for the Bhartiya Antariksh Hackathon 2024.