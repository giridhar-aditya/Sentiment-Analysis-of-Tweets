# test_model.py

import joblib
import re
import string

# Clean the input text
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)  # remove links
    text = re.sub(r"@\w+", "", text)      # remove mentions
    text = re.sub(r"#\w+", "", text)      # remove hashtags
    text = re.sub(rf"[{string.punctuation}]", "", text)  # remove punctuation
    text = text.strip()
    return text

# Load the model
def load_model():
    print("[INFO] Loading model...")
    model = joblib.load('sentiment_model.pkl')
    print("[INFO] Model loaded successfully.\n")
    return model

# Predict the sentiment
def predict_sentiment(model, text):
    cleaned_text = clean_text(text)
    prediction = model.predict([cleaned_text])[0]
    sentiment = "Positive" if prediction == 4 else "Negative"
    return sentiment

if __name__ == "__main__":
    print("======= SASTRA Sentiment Model Tester =======\n")
    
    model = load_model()

    print("Enter your text below (type 'exit' to quit):\n")

    while True:
        user_input = input("> ")
        if user_input.lower() == 'exit':
            print("\n[INFO] Exiting. Thank you for using the tester!")
            break

        sentiment = predict_sentiment(model, user_input)
        print(f"[Sentiment] {sentiment}\n")
