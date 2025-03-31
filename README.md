# 🎬 Fake News Detector and Explainer

## 📄 Project Overview

This project focuses on building a Fake News Detector model to classify fake and real news. Using Natural Language Processing (NLP) techniques, we analyze the textual data to predict the sentiment behind movie reviews. The concepts covered in this project form the basic building-block for understanding more complex Generative AI models, whose implementation will be covered in other projects.

## 📂 Description

The dataset used in this project is the IMDb Movie Reviews Dataset, sourced from Kaggle. This dataset is widely used for natural language processing (NLP) tasks, particularly for sentiment analysis. It contains a large collection of movie reviews from IMDb, along with their associated sentiment labels. The dataset can be downloaded from the following source:

Kaggle: https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews

### Dataset Summary

Total Reviews: 50,000

Labels: Binary (positive or negative sentiment)

Data Split: 25,000 reviews for training and 25,000 reviews for testing

Review Format: Text data containing user-submitted reviews

Sentiment Labels:

1 (positive sentiment)

0 (negative sentiment)

### Column Information

review: The text content of the movie review.

sentiment: The sentiment label, where 1 indicates a positive review and 0 indicates a negative review.

Source
This dataset was originally provided by Stanford University for use in sentiment classification tasks and was made available on Kaggle for broader usage in NLP research and practice.

Why This Dataset?
This dataset is ideal for sentiment analysis projects because:

It provides a balanced set of positive and negative reviews, making it perfect for training machine learning models.
The reviews are varied in length and vocabulary, providing a realistic challenge for text processing and model training.

## 🚀 Objectives

-Extract insights from IMDb movie reviews and determine if the sentiment is positive or negative.

-Implement various text preprocessing techniques to clean and prepare the data.

-Build and evaluate models using different feature extraction methods like Bag-of-Words and TF-IDF.

-Visualize the data distribution, word frequencies, and model performance using matplotlib and seaborn.

## 📊 Key Features

-Data Cleaning & Preprocessing: Removal of noise, punctuation, stopwords, and tokenization.

-Feature Engineering: Implementation of Bag-of-Words and TF-IDF to transform text into numerical features.

-Model Training: Trained a Logistic Regression classifier to predict the sentiment of IMDb reviews.

-Data Visualization: Visualized class distribution, review length, word clouds, n-grams, and model performance (confusion matrix, precision-recall, and ROC curves).

## 🛠️ Tools & Technologies

Python: Core programming language.

Jupyter Notebook: Development environment.

### Libraries:
pandas, numpy: Data manipulation and analysis.

scikit-learn: Model building, feature extraction, and evaluation.

matplotlib, seaborn: Data visualization.

nltk: Natural Language Processing toolkit.

wordcloud: Visualization of frequent words.

## 📉 Exploratory Data Analysis (EDA)

The following steps as part of the EDA were performed to understand the dataset:

-Class Distribution: Visualized the balance between positive and negative reviews.

-Word Clouds: Created separate word clouds for positive and negative reviews to highlight common words.

-TF-IDF Analysis: Visualized the most significant words based on their TF-IDF scores.

## 🧑‍💻 Model Building & Evaluation

Preprocessed the data using Bag-of-Words and TF-IDF vectorizers.

Trained a Logistic Regression model to classify reviews as positive or negative.

Evaluated model performance using:

-Confusion Matrix

-Precision-Recall Curve

-ROC Curve

## 📈 Results

The Logistic Regression model achieved:

Accuracy: 88%

Precision: 87%

Recall: 89%

F1-Score: 88%

## 🚀 Deployment

The trained Logistic Regression model was deployed as a web application using Streamlit. This allows users to input their own IMDb movie reviews and receive real-time sentiment predictions.

**Streamlit App Link:** https://sentiment-analysis-on-imdb-reviews-7zkvymoemaji92rpqqokxk.streamlit.app

**Key Features of the Streamlit App:**

-   Simple and user-friendly interface for inputting movie reviews.
-   Real-time sentiment prediction (positive or negative).
-   Clear display of the predicted sentiment.

**How to Use:**

1.  Visit the Streamlit app link.
2.  Enter your IMDb movie review in the provided text area.
3.  Click the "Analyze Sentiment" button.
4.  View the predicted sentiment displayed on the screen.

## 🛍️ Real-World Application: Product Review Analysis

Beyond movie reviews, this sentiment analysis model can be effectively applied to understand customer feedback on e-commerce platforms like Amazon and Alibaba.

**Use Case: Product Performance Analysis**

By analyzing product reviews, businesses can:

-   **Identify key product strengths and weaknesses:** Determine what customers like or dislike about a product.
-   **Monitor customer satisfaction:** Track changes in sentiment over time to gauge customer happiness.
-   **Gain insights into product quality:** Understand common issues and areas for improvement.
-   **Inform marketing and product development:** Tailor strategies based on customer feedback.

  This deployment showcases the versatile practical application of the sentiment analysis model, making it accessible for real-world use.
