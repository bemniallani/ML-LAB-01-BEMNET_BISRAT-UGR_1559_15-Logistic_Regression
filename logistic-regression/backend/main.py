from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import numpy as np
import os
app = FastAPI(title="Logistic Regression Diabetes Predictor")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "logistic_regression.joblib")
# Load the Logistic Regression model
try:
    model = joblib.load(MODEL_PATH)
    print("✅ Logistic Regression model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")

# Request model
class PatientData(BaseModel):
    Pregnancies: float
    Glucose: float
    BloodPressure: float
    SkinThickness: float
    Insulin: float
    BMI: float
    DiabetesPedigreeFunction: float
    Age: float

# Root endpoint
@app.get("/")
def home():
    return {
        "message": "Logistic Regression Diabetes Prediction API",
        "status": "active",
        "model": "Logistic Regression Classifier",
        "description": "Predicts diabetes risk using logistic regression"
    }

@app.get("/health")
def health_check():
    return {"status": "healthy", "model_loaded": True}

# Prediction endpoint
@app.post("/predict")
def predict(data: PatientData):
    try:
        # Prepare features array
        features = np.array([[
            data.Pregnancies,
            data.Glucose,
            data.BloodPressure,
            data.SkinThickness,
            data.Insulin,
            data.BMI,
            data.DiabetesPedigreeFunction,
            data.Age
        ]])
        
        # Get prediction and probabilities
        prediction = model.predict(features)[0]
        probabilities = model.predict_proba(features)[0]
        
        # Calculate risk score
        risk_score = float(probabilities[1] * 100)
        
        # Determine risk category
        if risk_score >= 70:
            risk_category = "High Risk"
            recommendation = "Consult a healthcare provider immediately"
        elif risk_score >= 40:
            risk_category = "Moderate Risk"
            recommendation = "Consider lifestyle changes and regular monitoring"
        else:
            risk_category = "Low Risk"
            recommendation = "Maintain healthy lifestyle"
        
        return {
            "prediction": int(prediction),
            "label": "Diabetic" if prediction == 1 else "Non-Diabetic",
            "probability_diabetic": float(probabilities[1]),
            "probability_non_diabetic": float(probabilities[0]),
            "risk_score": risk_score,
            "risk_category": risk_category,
            "recommendation": recommendation,
            "model": "logistic_regression",
            "features_used": list(data.dict().keys())
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Get feature importance
@app.get("/feature_importance")
def feature_importance():
    try:
        if hasattr(model.named_steps['classifier'], 'coef_'):
            coefficients = model.named_steps['classifier'].coef_[0]
            features = [
                "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
                "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"
            ]
            
            importance = {feature: abs(coef) for feature, coef in zip(features, coefficients)}
            sorted_importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))
            
            return {
                "feature_importance": sorted_importance,
                "message": "Higher absolute values indicate stronger influence"
            }
    except:
        return {"message": "Feature importance not available for this model configuration"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001) 