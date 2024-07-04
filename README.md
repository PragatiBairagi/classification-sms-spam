SMS Spam Detection
A machine learning project for classifying SMS messages as "Spam" or "Not Spam." The app uses text preprocessing, TF-IDF vectorization, and Multinomial Naive Bayes for spam detection. Built with Python’s scikit-learn, NLTK, and Streamlit for an interactive web interface.

Features
Text Processing: Tokenization, stopword removal, and stemming.
Model Training: Multinomial Naive Bayes for spam classification.
Web App: Streamlit-based interface for real-time predictions.
Installation

bash

git clone https://github.com/PragatiBairagi/sms-spam-classification.git
cd sms-spam-detection
pip install -r requirements.txt

Usage
Train Model: python train_model.py
Run App: streamlit run app.py
