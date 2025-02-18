import os
import pandas as pd
import json
import pickle
from sklearn.metrics import mean_squared_error, r2_score

def main():
    data_dir = "dvc/data/processed"
    models_dir = "dvc/models"
    metrics_dir = "dvc/metrics"

    # Ensure the metrics directory exists
    if not os.path.exists(metrics_dir):
        os.makedirs(metrics_dir)

    # Load test data
    X_test = pd.read_csv(os.path.join(data_dir, "X_test_scaled.csv"))
    y_test = pd.read_csv(os.path.join(data_dir, "y_test.csv")).squeeze()

    # Load the trained model
    model_path = os.path.join(models_dir, "gbr_model.pkl")
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    # Generate predictions
    predictions = model.predict(X_test)
    predictions_df = pd.DataFrame(predictions, columns=["predictions"])
    predictions_csv_path = "dvc/metrics/predictions.csv"
    predictions_df.to_csv(predictions_csv_path, index=False)
    print("Predictions saved to", predictions_csv_path)

    # Compute evaluation metrics
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    scores = {
        "MSE": mse,
        "R2": r2
    }
    
    # Save metrics to scores.json
    scores_path = os.path.join(metrics_dir, "scores.json")
    with open(scores_path, "w") as f:
        json.dump(scores, f, indent=4)
    print("Evaluation metrics saved to", scores_path)

if __name__ == "__main__":
    main()
