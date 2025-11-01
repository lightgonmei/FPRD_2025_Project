"""
src/prediction.py

Load the saved pipeline model and provide a predict_review() helper to clean
and predict a single review string, returning label and confidence.
"""

import os
import re
import joblib
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Ensure NLTK resources (quiet)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

STOP_WORDS = set(stopwords.words('english'))
LEMMATIZER = WordNetLemmatizer()

# Model path consistent with training and app (use 'models/' folder)
MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'fake_review_model.pkl')


def _clean_text_single(text: str) -> str:
    """Same cleaning logic used during preprocessing (keeps consistency)."""
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = [LEMMATIZER.lemmatize(w) for w in text.split() if w not in STOP_WORDS]
    return ' '.join(tokens)


def predict_review(review_text: str, model_path: str = MODEL_PATH):
    """
    Predict label and confidence for a single review string.
    Returns: (label_int, label_name, confidence_float)
    label_int: 0 or 1
    label_name: "Real" or "Fake"
    confidence_float: probability of the predicted class (0.0 - 1.0)
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}. Train the model first using model_training.train_and_save_model().")

    pipeline = joblib.load(model_path)
    cleaned = _clean_text_single(review_text)

    # Predict
    pred = pipeline.predict([cleaned])[0]
    if hasattr(pipeline, 'predict_proba'):
        proba = pipeline.predict_proba([cleaned])[0]
        confidence = float(proba[pred])
    else:
        # fallback: no probability support
        confidence = 1.0

    label_name = "Real" if int(pred) == 0 else "Fake"
    return int(pred), label_name, confidence
