# train_model.py

import pandas as pd
import re
import string
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# 1. Load data
def load_data(filepath):
    print("[INFO] Loading dataset...")
    df = pd.read_csv(filepath, header=None, encoding='latin-1')
    df.columns = ['target', 'id', 'date', 'query', 'user', 'text']
    print(f"[INFO] Dataset loaded with {len(df)} rows.")
    return df[['text', 'target']]

# 2. Clean text
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)  # remove links
    text = re.sub(r"@\w+", "", text)      # remove mentions
    text = re.sub(r"#\w+", "", text)      # remove hashtags
    text = re.sub(rf"[{string.punctuation}]", "", text)  # remove punctuations
    text = text.strip()
    return text

# 3. Train and save model
def train_and_save_model():
    print("[STEP 1] Loading and preparing data...")
    data = load_data('data.csv')

    print("[STEP 2] Cleaning text data...")
    data['text'] = data['text'].apply(clean_text)
    print("[INFO] Text cleaning completed.")

    print("[STEP 3] Splitting data into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        data['text'], data['target'], test_size=0.2, random_state=42
    )
    print(f"[INFO] Training set size: {len(X_train)} samples")
    print(f"[INFO] Test set size: {len(X_test)} samples")

    print("[STEP 4] Building the model pipeline...")
    model = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=5000)),
        ('clf', LogisticRegression(max_iter=1000))
    ])
    print("[INFO] Model pipeline created.")

    print("[STEP 5] Training the model...")
    model.fit(X_train, y_train)
    print("[INFO] Model training completed.")

    print("[STEP 6] Saving the model to 'sentiment_model.pkl'...")
    joblib.dump(model, 'sentiment_model.pkl')
    print("[SUCCESS] Model saved successfully!")

if __name__ == "__main__":
    print("======== SASTRA Sentiment Model Trainer ========")
    train_and_save_model()
    print("======== Training Finished ========")
