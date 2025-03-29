# -*- coding: utf-8 -*-
"""FND_streamlit app.ipynb"""
# Loading libraries
import streamlit as st
import tensorflow as tf
import numpy as np
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re
import nltk
from nltk.corpus import stopwords
from mistralai import Mistral

# Downloading necessary NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')

# Initializing NLTK
stop_words = set(stopwords.words("english"))
lemmatizer = nltk.WordNetLemmatizer()

# Custom CSS for styling
st.set_page_config(page_title="Fake News Detector & Explainer", layout="wide")
st.markdown("""
<style>
    .reportview-container {
        background: #4A4A4A
    }
    .big-font {
        font-size:20px !important;
    }
    .result-box {
        background-color: #4A4A4A;
        color: white;
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
        
    return lstm_model, lstm_tokenizer

lstm_model, lstm_tokenizer = load_models()

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

def generate_mistral_explanation(text, prediction, lstm_proba):
    api_key =   "tNsDSmxLXwubltb5tdZkOdkOvqVLv56r" # Use Streamlit secrets
    client = Mistral(api_key=api_key)
    prompt = f"""The following news article was classified as {prediction} with {float(1-lstm_proba):.2f}% confidence.\n\n{text}\n\nExplain in simple terms why this news might be classified as {prediction}. Fact check the classification."""
    messages = [{"role":"system", "content":"You are an AI expert in fake news detection."},
        {"role":"user", "content":prompt}
    ]
    try:
        response = client.chat.complete(model="mistral-medium", messages=messages)
        return response.choices[0].message.content
    except Exception as e:
        return f"Error generating explanation: {e}"

def analyze_news(text):
    cleaned_text = clean_text(text)
    sequence = lstm_tokenizer.texts_to_sequences([cleaned_text])
    padded = pad_sequences(sequence, maxlen=100, padding='post')
    lstm_proba = lstm_model.predict(padded, verbose=0)[0][0]
    prediction = "Fake" if lstm_proba > 0.5 else "Real"
    explanation = generate_mistral_explanation(text, prediction, lstm_proba)
    return {
        "prediction": prediction,
        "confidence": float(lstm_proba if prediction == "Fake" else 1 - lstm_proba),
        "explanation": explanation,
    }

# Streamlit UI
st.title("Fake News Detector & Explainer")
st.markdown("""
This tool combines:
- **LSTM** for classification
- **Mistral API** for explanation
""")

news_input = st.text_area("Paste news article or headline:", height=150)

if st.button("Analyze"):
    if not news_input.strip():
        st.warning("Please enter some text!")
    else:
        with st.spinner("Analyzing..."):
            result = analyze_news(news_input)

        # Display results
        st.markdown(f"""  
        <div class="result-box">
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
    Model: LSTM + Mistral API | Made with Streamlit
</div>
""", unsafe_allow_html=True)

