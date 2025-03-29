# -*- coding: utf-8 -*-
"""FND_streamlit app.ipynb"""
# Loading libraries
import requests
import os
import tempfile
import streamlit as st
import tensorflow as tf
from transformers import T5Tokenizer, T5ForConditionalGeneration
import sentencepiece
import numpy as np
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re
import nltk
from nltk.corpus import stopwords
import torch
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage

# Downloading necessary NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')

# Initializing NLTK
stop_words = set(stopwords.words("english"))
lemmatizer = nltk.WordNetLemmatizer()

# Custom CSS for styling
st.set_page_config(page_title="Fake News Detector", layout="wide")
st.markdown("""
<style>
    .reportview-container {
        background: #f0f2f6
    }
    .big-font {
        font-size:20px !important;
    }
    .result-box {
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0px;
    }
    .fake {
        background-color: #ffdddd;
    }
    .real {
        background-color: #ddffdd;
    }
</style>
""", unsafe_allow_html=True)

# Loading models and caching them to prevent reloading
@st.cache_resource
def load_models():
    # LSTM Model
    lstm_model = tf.keras.models.load_model("Models/LSTM Model/Fake_News_Detector_Model.keras")
    with open("Models/LSTM Model/tokenizer.pkl", 'rb') as f:
        lstm_tokenizer = pickle.load(f)

    # T5 Model
    tokenizer = T5Tokenizer.from_pretrained('t5-small')
    model = T5ForConditionalGeneration.from_pretrained('t5-small')
    model.eval()
    return lstm_model, lstm_tokenizer, model, tokenizer

lstm_model, lstm_tokenizer, t5_model, t5_tokenizer = load_models()

# Function to clean the input
def clean_text(text):
    if isinstance(text, str):
        text = text.lower()  # Converting to lowercase
        text = re.sub(r"<.*?>", "", text)  # Removing HTML tags
        text = re.sub(r"http\S+|www\S+", "", text)  # Removing URLs
        text = re.sub(r"[:,!?]", "", text) # Removing colons, commas, question marks and exclamation marks
        text = re.sub(r"[^a-zA-Z0-9\s]", "", text) #Removing all other special characters
        text = " ".join([lemmatizer.lemmatize(word) for word in text.split() if word not in stop_words])  # Lemmatization & stopword removal
        return text
    return ""

def analyze_news(text):
    # LSTM prediction
    cleaned_text = clean_text(text)
    sequence = lstm_tokenizer.texts_to_sequences([cleaned_text])
    padded = pad_sequences(sequence, maxlen=100, padding='post')
    lstm_proba = lstm_model.predict(padded, verbose=0)[0][0]
    prediction = "Fake" if lstm_proba > 0.5 else "Real"

    api_key = "tNsDSmxLXwubltb5tdZkOdkOvqVLv56r"  # Replace with your actual API key
    model = "mistral-medium"  # Or another Mistral model of your choice
    client = MistralClient(api_key=api_key)

    messages = [
        ChatMessage(role="user", content=f"Explain why {text} is {prediction}.")
    ]
    chat_response = client.chat(model=model, messages=messages)
    explanation = chat_response.choices[0].message.content
    return explanation

# Streamlit UI
st.title("🔍 Fake News Detector")
st.markdown("""
This tool combines:
- **LSTM** for classification
- **T5-Small Model** for explanation
""")

news_input = st.text_area("Paste news article or headline:", height=150)

if st.button("Analyze"):
    if not news_input.strip():
        st.warning("Please enter some text!")
    else:
        with st.spinner("Analyzing..."):
            result = analyze_news(news_input)

        # Display results
        box_class = "fake" if result["prediction"] == "Fake" else "real"
        st.markdown(f"""
        <div class="result-box {box_class}">
            <h3>Result: <span style='color: {"red" if result["prediction"] == "Fake" else "green"}'>{result["prediction"]}</span></h3>
            <p class="big-font">Confidence: <b>{result['confidence']:.1%}</b></p>
            <p><b>Explanation:</b> {result['explanation']}</p>
        </div>
        """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<style>
    .footer {
        text-align: center;
        padding: 10px;
        color: gray;
    }
</style>
<div class="footer">
    Model: LSTM + T5-Small | Made with Streamlit
</div>
""", unsafe_allow_html=True)

