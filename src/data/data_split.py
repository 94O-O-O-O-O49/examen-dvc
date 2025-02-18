import os
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

def main():
    # Load parameters from params.yaml
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)
    split_params = params.get("split", {})
    test_size = split_params.get("test_size", 0.2)
    random_state = split_params.get("random_state", 42)

    # Load raw data
    raw_data_path = "examen-dvc/data/raw/raw.csv"
    df = pd.read_csv(raw_data_path)

    # Drop the 'date' column if it exists
    if "date" in df.columns:
        df = df.drop("date", axis=1)

    # Separate features and target
    target = "silica_concentrate"
    X = df.drop(target, axis=1)
    y = df[target]

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Ensure output directory exists
    output_dir = "examen-dvc/data/processed"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Save outputs
    X_train.to_csv(os.path.join(output_dir, "X_train.csv"), index=False)
    X_test.to_csv(os.path.join(output_dir, "X_test.csv"), index=False)
    y_train.to_csv(os.path.join(output_dir, "y_train.csv"), index=False)
    y_test.to_csv(os.path.join(output_dir, "y_test.csv"), index=False)
    print("Data splitting completed and files saved in", output_dir)

if __name__ == "__main__":
    main()
