import os
import pandas as pd
from src.data_preprocessing import load_and_preprocess
from src.model_training import train_and_save_model
from src.evaluation_metrics import evaluate_and_plot

# Paths
DATA_PATH = "dataset/Electronics_Products_Dataset.csv"
MODEL_DIR = "models"

def main():
    print("🔹 Step 1: Loading dataset...")
    df = pd.read_csv(DATA_PATH)
    print(f"✅ Dataset loaded successfully with {len(df)} rows.")

    print("\n🔹 Step 2: Preprocessing data...")
    X_train, X_test, y_train, y_test = load_and_preprocess(DATA_PATH)
    print("✅ Data preprocessing completed.")

    print("\n🔹 Step 3: Training model (this will save to the project 'models/' folder)...")
    pipeline = train_and_save_model(X_train, y_train)
    print("✅ Model training completed and saved.")

    print("\n🔹 Step 4: Evaluating model...")
    evaluate_and_plot(pipeline, X_test, y_test)
    print("✅ Evaluation completed.")

    print("\n🎯 Training pipeline completed successfully!")


if __name__ == "__main__":
    main()
