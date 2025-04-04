# Fake News Detector and Explainer

## Project Overview

This project focuses on building a Fake News Detector model to classify fake and real news. In today's globally connected world, social media plays an important role in everyday life. It is no longer a luxury but rather a necessity without which it is hard to imagine the existence of the modern world. However, this interconnectedness also facilitates the rapid spread of misinformation, making it crucial to develop tools that can identify and mitigate the impact of fake news. This project aims to address this challenge by creating a robust fake news detection system that not only classifies news articles and headlines but also provides explanations for its predictions and fact-checks them.

## Description

To build a robust fake news detection system, this project employs a multi-source dataset, combining Twitter news dataset, samples from Gossipcop and Politifact datasets and synthetically generated data. After thorough cleaning and preprocessing, an LSTM model was trained to classify news articles as real or fake. The model's predictions are then passed to the  `mistral-medium` model through an API key, which generates comprehensive explanations and performs fact-checking, providing a layer of validation and insight into the model's reasoning.

Upcoming planned enhancements include implementing a feedback loop that utilizes the Mistral API's fact-checking results for model retraining, optimizing its classification accuracy.

### Dataset Summary

-   Total Samples: 77716
-   Labels: Binary (fake news: 1, real news: 0)

To enhance the model's performance and address class imbalance some data was synthetically augmented using the `faker` library. The augmented data consisted synthetically generated fake news from various domains such as geopolitics, technology, science, history, culture, past, future, health, entertainment, sports, finance, environment, space exploration, artificial intelligence, social media, climate change and education.
The fake news created using the `faker` library, produced realistic-sounding but fictional data, which generated coherent and contextually relevant text. This approach ensured a diverse and challenging dataset that mirrors the complexities of real-world fake news.

## Results

LSTM Model
-   Accuracy: 94.54 %
-   Precision: 96.39 %
-   Recall: 92.54 %
-   F1 Score: 94.43 %

## Deployment

The trained LSTM model and explainer were deployed as a web application using Streamlit. This allows users to input news articles and receive real-time fake news predictions along with fact-checking explanations.

**Streamlit App Link:** https://fake-news-detector-and-explainer-8bgvqtt9mnhq5b6yf5fubt.streamlit.app

**How to Use:**

1.  Visit the Streamlit app link.
2.  Enter the news article text in the provided text area.
3.  Click the "Analyze" button.
4.  View the predicted label and the fact-checking explanation.
