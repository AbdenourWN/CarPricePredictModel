import joblib
import pandas as pd

from custom_transformers import TargetEncoder

def predict_price(car_data: dict) -> float:
    """
    Loads the trained model pipeline and makes a price prediction.
    Args:
        car_data (dict): A dictionary containing the car's features.
    Returns:
        float: The predicted price of the car.
    """
    try:
        # Load the pre-trained pipeline
        pipeline = joblib.load("champion_car_price_pipeline.pkl")
    except FileNotFoundError:
        print("Error: Model file 'champion_car_price_pipeline.pkl' not found.")
        print("Please run the 'train.py' script first to create the model file.")
        return None

    # Convert the input dictionary to a pandas DataFrame
    input_df = pd.DataFrame([car_data])

    # Use the pipeline to make a prediction
    predicted_price = pipeline.predict(input_df)

    return float(predicted_price[0])

if __name__ == '__main__':
    # This is an example of how to use the function.
    # Your FastAPI app would call the predict_price function directly.
    sample_car = {
        'marque': 'Mazda',
        'modele': '2',
        'puissance_fiscale': 5,
        'kilometrage': 197000,
        'annee': 2014,
        'energie': 'Essence',
        'boite': 'Manuelle'
    }

    print("Making a prediction for a sample car:")
    print(sample_car)
    
    price = predict_price(sample_car)

    if price:
        print(f"\n---> Predicted Price: {price:,.2f}")