"""
src/data_preprocessing.py

Loads the dataset (dataset/Electronics_Products_Dataset.csv),
performs advanced cleaning (stopword removal + lemmatization),
filters very short reviews, and returns train/test splits.
"""

import os
import re
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split

# Ensure NLTK data (downloads quietly if needed)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

STOP_WORDS = set(stopwords.words('english'))
LEMMATIZER = WordNetLemmatizer()


def _clean_text_single(text: str) -> str:
    """
    Clean a single text string:
    - lowercasing
    - remove URLs and HTML tags
    - remove non-alphabetic characters
    - remove extra whitespace
    - remove stopwords and lemmatize
    """
    text = str(text).lower()
    # Remove URLs
    text = re.sub(r'http\S+|www\S+', '', text)
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    # Keep only letters and spaces
    text = re.sub(r'[^a-z\s]', ' ', text)
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    # Tokenize, remove stopwords, lemmatize
    tokens = [LEMMATIZER.lemmatize(w) for w in text.split() if w not in STOP_WORDS]
    return ' '.join(tokens)


def load_and_preprocess(file_path: str,
                        text_col: str = 'text_',
                        label_col: str = 'label',
                        test_size: float = 0.2,
                        random_state: int = 42,
                        min_words: int = 3):
    """
    Loads CSV, keeps only text_col and label_col, cleans text, filters short rows,
    and returns X_train, X_test, y_train, y_test (pandas Series).
    Accepts files separated by commas or tabs.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset not found: {file_path}")

    # Robust CSV load with proper quoting and escaping
    try:
        # First try comma-separated with quote handling
        df = pd.read_csv(file_path, 
                        sep=',',
                        quoting=1,  # QUOTE_ALL
                        escapechar='\\',
                        on_bad_lines='warn')
    except Exception as e:
        print(f"Warning: First CSV read attempt failed, trying alternative format... ({str(e)})")
        try:
            # Try tab-separated if comma fails
            df = pd.read_csv(file_path,
                           sep='\t',
                           quoting=1,  # QUOTE_ALL
                           escapechar='\\',
                           on_bad_lines='warn')
        except Exception as e:
            print(f"Warning: Second CSV read attempt failed, trying minimal options... ({str(e)})")
            # Final attempt with minimal options
            df = pd.read_csv(file_path, on_bad_lines='skip')
            print("Success: Loaded CSV with minimal options (some malformed lines may have been skipped)")

    # Ensure required columns exist
    if text_col not in df.columns or label_col not in df.columns:
        raise KeyError(f"Required columns not found. Expected '{text_col}' and '{label_col}' in dataset.")

    df = df[[text_col, label_col]].dropna().copy()

    # Clean text (vectorized apply)
    print("🧹 Cleaning text (this may take a moment)...")
    df[text_col] = df[text_col].astype(str).apply(_clean_text_single)

    # Remove very short reviews
    df['__len__'] = df[text_col].str.split().apply(len)
    df = df[df['__len__'] >= min_words].drop(columns='__len__')

    # Ensure labels are integers (0/1)
    df[label_col] = df[label_col].astype(int)

    # Train/test split
    X = df[text_col]
    y = df[label_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    print(f"✅ Loaded {len(df)} samples (train={len(X_train)}, test={len(X_test)})")
    return X_train.reset_index(drop=True), X_test.reset_index(drop=True), y_train.reset_index(drop=True), y_test.reset_index(drop=True)
