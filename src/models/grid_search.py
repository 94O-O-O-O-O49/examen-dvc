import os
import pandas as pd
import yaml
import pickle
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV

def main():
    # Load grid search parameters from params.yaml
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)
    grid_params = params.get("grid_search", {})

    # Define parameter grid (example defaults provided)
    param_grid = {
        "n_estimators": grid_params.get("n_estimators", [100]),
        "learning_rate": grid_params.get("learning_rate", [0.1]),
        "max_depth": grid_params.get("max_depth", [3]),
    }
    
    # Load training data
    data_dir = "examen-dvc/data/processed"
    X_train = pd.read_csv(os.path.join(data_dir, "X_train_scaled.csv"))
    y_train = pd.read_csv(os.path.join(data_dir, "y_train.csv")).squeeze()

    # Initialize the regressor and perform grid search
    model = GradientBoostingRegressor()
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring="neg_mean_squared_error")
    grid_search.fit(X_train, y_train)
    
    # Ensure models directory exists
    output_dir = "examen-dvc/models"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # Save best parameters to file
    best_params_path = os.path.join(output_dir, "best_params.pkl")
    with open(best_params_path, "wb") as f:
        pickle.dump(grid_search.best_params_, f)
    print("Grid search complete. Best parameters saved to", best_params_path)

if __name__ == "__main__":
    main()
