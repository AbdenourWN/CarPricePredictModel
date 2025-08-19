# main.py
import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

# --- IMPORT OUR CUSTOM CLASS ---
# This is crucial. The script needs the definition of TargetEncoder
# to be able to load the pipeline file.
from custom_transformers import TargetEncoder


# 1. Initialize the FastAPI application
app = FastAPI(title="Car Price Prediction API", version="1.0")


# 2. Add CORS Middleware
# This is a security feature that allows your frontend application (running on a different domain)
# to make requests to this API. For a portfolio project, allowing all origins is fine.
# For a real production app, you would restrict this to your frontend's domain.
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# 3. Load the pre-trained model pipeline
# This is done once when the application starts up, making predictions fast.
try:
    pipeline = joblib.load("champion_car_price_pipeline.pkl")
    print("Model loaded successfully!")
except FileNotFoundError:
    print("Error: Model file 'champion_car_price_pipeline.pkl' not found.")
    pipeline = None


# 4. Define the input data model using Pydantic
# FastAPI will use this for automatic data validation, conversion, and documentation.
# The frontend must send a JSON object with these exact field names.
class CarData(BaseModel):
    marque: str
    modele: str
    puissance_fiscale: int
    kilometrage: int
    annee: int
    energie: str
    boite: str


# 5. Create the root endpoint (for health checks)
@app.get("/")
def read_root():
    return {"status": "ok", "message": "Welcome to the Car Price Prediction API!"}


# 6. Create the prediction endpoint
@app.post("/predict")
def predict_price(car: CarData):
    """
    Takes car features as input and returns the predicted price.
    """
    if pipeline is None:
        return {"error": "Model is not loaded. Please check the server logs."}
    
    # Convert the input Pydantic model to a dictionary
    data = car.dict()
    
    # Convert the dictionary to a pandas DataFrame
    # The pipeline expects a DataFrame as input.
    input_df = pd.DataFrame([data])
    
    # Use the pipeline to make a prediction
    # The pipeline handles all preprocessing steps internally.
    prediction = pipeline.predict(input_df)
    
    # Return the prediction in a JSON response
    # We cast to float to ensure it's JSON serializable.
    return {"predicted_price": float(prediction[0])}