import streamlit as st
import tensorflow as tf
import pickle
import re
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Set page configuration
st.set_page_config(page_title="Fake News Detection", page_icon="📰", layout="centered")

# Load model
model = tf.keras.models.load_model("fake_news_lstm_model.keras")

# Load tokenizer
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# Define constants
max_length = 500

# Clean input text
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

# App title
st.title("📰 Fake News Detection")
st.write("Enter a news article or headline to check whether it is likely real or fake.")

# Text input
user_input = st.text_area("Paste the news text here:")

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        # Clean and preprocess input text
        cleaned_text = clean_text(user_input)
        sequence = tokenizer.texts_to_sequences([cleaned_text])
        padded_sequence = pad_sequences(sequence, maxlen=max_length, padding="post")

        # Make prediction
        prediction = model.predict(padded_sequence)[0][0]

        # Display result
        if prediction >= 0.5:
            st.success(f"This news is likely REAL. Confidence: {prediction:.2f}")
        else:
            st.error(f"This news is likely FAKE. Confidence: {1 - prediction:.2f}")