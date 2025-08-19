import pandas as pd
import joblib
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import r2_score, mean_squared_error

import xgboost as xgb

# --- IMPORT OUR CUSTOM CLASS ---
from custom_transformers import TargetEncoder

def train_model():
    """
    Main function to load data, train the model pipeline, and save it.
    """
    print("--- Car Price Prediction: Training the Final Champion Model ---")

    # --- 1. DATA PREPARATION ---
    print("Step 1: Loading, cleaning, and preparing data...")
    df = pd.read_csv('car_dataset.csv')
    df.dropna(subset=['prix'], inplace=True)
    for col in ['puissance_fiscale', 'kilometrage', 'annee']:
        if df[col].isnull().any():
            df[col].fillna(df[col].median(), inplace=True)
    for col in ['marque', 'modele', 'energie', 'boite']:
        if df[col].isnull().any():
            df[col].fillna(df[col].mode()[0], inplace=True)
    price_cap = df['prix'].quantile(0.99)
    price_floor = df['prix'].quantile(0.01)
    df = df[(df['prix'] < price_cap) & (df['prix'] > price_floor)]
    X = df.drop('prix', axis=1)
    y = df['prix']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # --- 2. PREPROCESSING PIPELINE ---
    print("Step 2: Defining the preprocessing pipeline...")
    numerical_features = ['puissance_fiscale', 'kilometrage', 'annee']
    categorical_features = ['marque', 'energie', 'boite']
    high_cardinality_features = ['modele']

    preprocessor = ColumnTransformer(
        transformers=[
            ('target_encoder', TargetEncoder(), high_cardinality_features),
            ('numeric_scaler', StandardScaler(), numerical_features),
            ('one_hot_encoder', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ],
        remainder='passthrough'
    )

    # --- 3. CREATE AND TRAIN THE FINAL TUNED MODEL ---
    print("Step 3: Creating and training the final, tuned model...")
    best_params = {
        'subsample': 0.9, 'n_estimators': 300, 'max_depth': 7,
        'learning_rate': 0.05, 'gamma': 0.1, 'colsample_bytree': 0.7,
        'objective': 'reg:squarederror', 'random_state': 42, 'n_jobs': -1
    }

    final_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', xgb.XGBRegressor(**best_params))
    ])

    final_pipeline.fit(X_train, y_train)
    print("Champion model training complete.")

    # --- 4. EVALUATE THE FINAL MODEL ---
    print("\nEvaluating the final champion model on the test set...")
    y_pred = final_pipeline.predict(X_test)
    final_r2 = r2_score(y_test, y_pred)
    final_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print("\n--- Final Champion Model Evaluation ---")
    print(f"Final R-squared (R²): {final_r2:.4f}")
    print(f"Final RMSE: {final_rmse:.2f}")

    # --- 5. SAVE THE MODEL ---
    print("\nSaving the final champion pipeline to 'champion_car_price_pipeline.pkl'...")
    joblib.dump(final_pipeline, 'champion_car_price_pipeline.pkl')
    print("Champion model saved successfully!")
    print("--- Script Finished ---")


if __name__ == '__main__':
    train_model()