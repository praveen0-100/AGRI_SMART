import os
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

MODEL_PATH = os.path.join(os.path.dirname(__file__), "soil_classifier.pkl")

# Crop target profiles (N, P, K, pH, Moisture, Temp)
CROP_PROFILES = {
    "Rice": {"N": 100, "P": 50, "K": 50, "ph": 6.0, "moisture": 75, "temp": 28},
    "Groundnut": {"N": 30, "P": 50, "K": 30, "ph": 6.5, "moisture": 40, "temp": 26},
    "Maize": {"N": 120, "P": 60, "K": 40, "ph": 6.5, "moisture": 60, "temp": 24},
    "Sugarcane": {"N": 200, "P": 75, "K": 100, "ph": 6.8, "moisture": 70, "temp": 30},
    "Cotton": {"N": 80, "P": 40, "K": 40, "ph": 7.0, "moisture": 50, "temp": 28},
    "Jowar": {"N": 60, "P": 30, "K": 30, "ph": 6.8, "moisture": 30, "temp": 30},
    "Ragi": {"N": 50, "P": 35, "K": 25, "ph": 6.2, "moisture": 35, "temp": 25}
}

def train_and_save_model():
    """Generate synthetic agronomy data and train a RandomForest soil classifier."""
    print("Training soil ML model...")
    np.random.seed(42)
    data = []
    
    # Generate 300 samples for each crop with random variation
    for crop, profile in CROP_PROFILES.items():
        for _ in range(300):
            row = {
                "N": max(5.0, profile["N"] + np.random.normal(0, 15)),
                "P": max(5.0, profile["P"] + np.random.normal(0, 10)),
                "K": max(5.0, profile["K"] + np.random.normal(0, 10)),
                "ph": np.clip(profile["ph"] + np.random.normal(0, 0.4), 4.0, 9.5),
                "moisture": np.clip(profile["moisture"] + np.random.normal(0, 8), 10, 100),
                "temp": np.clip(profile["temp"] + np.random.normal(0, 3), 15, 45),
                "crop": crop
            }
            data.append(row)
            
    df = pd.DataFrame(data)
    X = df[["N", "P", "K", "ph", "moisture", "temp"]]
    y = df["crop"]
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    print("Model saved to:", MODEL_PATH)
    return model

def get_model():
    """Load or train-load the RandomForest model."""
    if not os.path.exists(MODEL_PATH):
        return train_and_save_model()
    try:
        with open(MODEL_PATH, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        print(f"Error loading soil model: {e}, retraining...")
        return train_and_save_model()

# Load the model
model = get_model()

def predict_soil_crop(n: float, p: float, k: float, ph: float, moisture: float, temp: float) -> dict:
    """Predict crop and calculate fertilizer discrepancies based on NPK defaults."""
    input_data = [[n, p, k, ph, moisture, temp]]
    
    # Get top predicted crop and probability list
    pred = model.predict(input_data)[0]
    probs = model.predict_proba(input_data)[0]
    classes = model.classes_
    
    # Sort predictions by probability
    ranked = sorted(zip(classes, probs), key=lambda x: x[1], reverse=True)
    
    # Get target profile for the top predicted crop
    profile = CROP_PROFILES.get(pred, CROP_PROFILES["Rice"])
    
    # Calculate fertilizer deficits
    def_n = max(0.0, profile["N"] - n)
    def_p = max(0.0, profile["P"] - p)
    def_k = max(0.0, profile["K"] - k)
    
    # Recommendations to supply missing nutrients:
    # 1. Urea contains 46% Nitrogen
    # 2. SSP contains 16% Phosphorus (P2O5)
    # 3. MOP contains 60% Potassium (K2O)
    urea_needed = def_n / 0.46
    ssp_needed = def_p / 0.16
    mop_needed = def_k / 0.60
    
    return {
        "recommended_crop": pred,
        "alternatives": [{"crop": c, "probability": float(pb)} for c, pb in ranked[1:4]],
        "nutrient_status": {
            "nitrogen_status": "optimal" if def_n == 0 else "deficient",
            "phosphorus_status": "optimal" if def_p == 0 else "deficient",
            "potassium_status": "optimal" if def_k == 0 else "deficient",
        },
        "fertilizer_plan_kg_per_ha": {
            "urea_nitrogen": round(urea_needed, 1),
            "single_superphosphate_phosphorus": round(ssp_needed, 1),
            "muriate_of_potash_potassium": round(mop_needed, 1),
            "farm_yard_manure": 12000.0  # standard organic baseline
        }
    }
