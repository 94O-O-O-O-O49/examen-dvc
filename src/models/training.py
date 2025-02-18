import os
import pandas as pd
import pickle
from sklearn.ensemble import GradientBoostingRegressor

def main():
    data_dir = "examen-dvc/data/processed"
    models_dir = "examen-dvc/models"

    # Load training data
    X_train = pd.read_csv(os.path.join(data_dir, "X_train_scaled.csv"))
    y_train = pd.read_csv(os.path.join(data_dir, "y_train.csv")).squeeze()

    # Load best parameters from grid search
    best_params_path = os.path.join(models_dir, "best_params.pkl")
    with open(best_params_path, "rb") as f:
        best_params = pickle.load(f)
    
    # Train the model using best parameters
    model = GradientBoostingRegressor(**best_params)
    model.fit(X_train, y_train)
    
    # Save the trained model
    model_path = os.path.join(models_dir, "gbr_model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    print("Model training complete. Model saved to", model_path)

if __name__ == "__main__":
    main()
