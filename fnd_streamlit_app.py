# -*- coding: utf-8 -*-
"""FND_streamlit app.ipynb"""

!pip install streamlit

# Loading libraries
import requests
import os
import streamlit as st
import tensorflow as tf
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch
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

def download_from_onedrive(share_link, save_path):
    """Download a file from OneDrive shared link"""
    # Convert sharing link to direct download URL
    direct_url = share_link.replace("share", "download") \
                         .replace("?shareKey=", "?download=1&shareKey=") \
                         .replace("?shareId=", "?download=1&shareId=")
    
    response = requests.get(direct_url)
    with open(save_path, 'wb') as f:
        f.write(response.content)

# Loading models and caching them to prevent reloading
@st.cache_resource
def load_models():
    # Create models directory if it doesn't exist
    os.makedirs("models", exist_ok=True)
    
    # Define OneDrive file URLs (replace with your actual links)
    ONEDRIVE_FILES = {
        "Fake_News_Detector_Model.h5": "https://1drv.ms/u/s!YOUR_LSTM_MODEL_LINK",
        "tokenizer.pkl": "https://1drv.ms/u/s!YOUR_TOKENIZER_LINK",
        "DistilBERT_model.pt": "https://1drv.ms/u/s!YOUR_DISTILBERT_MODEL_LINK",
        "distilbert_model/config.json": "https://1drv.ms/u/s!YOUR_CONFIG_LINK",
        "distilbert_model/tokenizer_config.json": "https://1drv.ms/u/s!YOUR_TOKENIZER_CONFIG_LINK",
        "distilbert_model/vocab.txt": "https://1drv.ms/u/s!YOUR_VOCAB_LINK"
    }

    # Download files if they don't exist locally
    for filepath, url in ONEDRIVE_FILES.items():
        if not os.path.exists(f"models/{filepath}"):
            os.makedirs(os.path.dirname(f"models/{filepath}"), exist_ok=True)
            download_from_onedrive(url, f"models/{filepath}")

    # LSTM Model
    lstm_model = tf.keras.models.load_model('/content/Fake_News_Detector_Model.h5')
    with open('/content/tokenizer.pkl', 'rb') as f:
        lstm_tokenizer = pickle.load(f)

    # DistilBERT Model
    tokenizer = DistilBertTokenizer.from_pretrained('/content/drive/MyDrive/Fake News/distilbert_model')
    model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased')  # Load architecture
    model = torch.quantization.quantize_dynamic(
        model,
        {torch.nn.Linear},
        dtype=torch.qint8
    )
    model.load_state_dict(torch.load('/content/DistilBERT_model.pt', map_location=torch.device('cpu')))  # Load weights
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
    explanation = distilbert_tokenizer.decode(outputs[0], skip_special_tokens=True).split(":")[-1].strip()

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

