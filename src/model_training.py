"""
src/model_training.py

Train a TF-IDF + MultinomialNB pipeline and save it as model/fake_review_model.pkl.
"""

import os
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import joblib

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
MODEL_PATH = os.path.join(MODEL_DIR, 'fake_review_model.pkl')


def train_and_save_model(X_train, y_train,
                         max_features: int = 7000,
                         ngram_range: tuple = (1, 2)):
    """
    Train pipeline and save to disk. Returns the trained pipeline.
    """
    os.makedirs(MODEL_DIR, exist_ok=True)

    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(
            max_features=max_features,
            stop_words='english',
            ngram_range=ngram_range)),
        ('clf', MultinomialNB())
    ])

    print("🧠 Training model pipeline (TF-IDF + MultinomialNB)...")
    pipeline.fit(X_train, y_train)

    joblib.dump(pipeline, MODEL_PATH)
    print(f"✅ Model saved to: {MODEL_PATH}")
    return pipeline


def load_model(model_path: str = MODEL_PATH):
    """Load the saved pipeline. Raises FileNotFoundError if missing."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}. Train the model first.")
    return joblib.load(model_path)
