import os
import pandas as pd
from sklearn.preprocessing import StandardScaler

def main():
    input_dir = "examen-dvc/data/processed"
    X_train_path = os.path.join(input_dir, "X_train.csv")
    X_test_path = os.path.join(input_dir, "X_test.csv")
    
    # Load datasets
    X_train = pd.read_csv(X_train_path)
    X_test = pd.read_csv(X_test_path)

    # Apply StandardScaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Convert back to DataFrame to preserve column names
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)

    # Save scaled datasets
    X_train_scaled.to_csv(os.path.join(input_dir, "X_train_scaled.csv"), index=False)
    X_test_scaled.to_csv(os.path.join(input_dir, "X_test_scaled.csv"), index=False)
    print("Feature normalization completed. Scaled files saved in", input_dir)

if __name__ == "__main__":
    main()
