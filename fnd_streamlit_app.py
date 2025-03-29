# -*- coding: utf-8 -*-
"""FND_streamlit app.ipynb"""

# Fixes at the very top
#import asyncio
#import sys
#if sys.platform == "win32" and (3, 8) <= sys.version_info < (3, 9):
    #asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

#import torch
#torch._C._disable_torch_functional_class_checks = True

# Loading libraries
import requests
import os
import tempfile
import joblib
import streamlit as st
import tensorflow as tf
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import numpy as np
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re
import nltk
from nltk.corpus import stopwords

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
    #lstm_layer = tf.keras.layers.TFSMLayer("Models/LSTM Model/Fake_News_Detector_Model.h5", call_endpoint='serving_default')
    #lstm_model = tf.keras.Sequential([lstm_layer])
    lstm_model = joblib.load("Models/LSTM Model/Fake_News_Detector_Model.h5")
    with open("Models/LSTM Model/tokenizer.pkl", 'rb') as f:
        lstm_tokenizer = pickle.load(f)

    # DistilBERT Model
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased')  # Load architecture
    #model.load_state_dict(torch.load(distilbert_model_path, map_location=torch.device('cpu')))  # Load weights
    model.eval()
    return lstm_model, lstm_tokenizer, model, tokenizer

lstm_model, lstm_tokenizer, distilbert_model, distilbert_tokenizer = load_models()

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

    # DistilBERT Explanation
    inputs = distilbert_tokenizer(
        f"Explain why this news is {prediction} in one or two sentences: {text}",
        return_tensors="pt",
        truncation=True,
        max_length=128
    )
    with torch.no_grad():
        outputs = distilbert_model.generate(**inputs, max_new_tokens=50)
    try:
        explanation = distilbert_tokenizer.decode(outputs[0], skip_special_tokens=True).split(":")[-1].strip()
    except IndexError:
        explanation = "Explanation could not be generated."
    except KeyError as e:
        explanation = f"Explanation generation error. Key error: {e}"
    except Exception as e:
        explanation = f"An unexpected error occurred during explanation generation: {e}"

    return {
        "prediction": prediction,
        "confidence": float(lstm_proba if prediction == "Fake" else 1 - lstm_proba),
        "explanation": explanation
    }

# Streamlit UI
st.title("🔍 Fake News Detector")
st.markdown("""
This tool combines:
- **LSTM** for classification
- **DistilBERT** for explanation
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
    Model: LSTM + DistilBERT | Made with Streamlit
</div>
""", unsafe_allow_html=True)

