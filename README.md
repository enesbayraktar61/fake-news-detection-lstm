# Fake News Detection (LSTM)

This project builds a binary text classification model to detect whether a news article is real or fake using deep learning.

The model was trained using TensorFlow and deployed with Streamlit.

---

## Project Overview

- **Problem Type:** Binary Text Classification  
- **Approach:** Deep Learning (LSTM)  
- **Framework:** TensorFlow / Keras  
- **Deployment:** Streamlit  

---

## Dataset

The dataset consists of labeled news articles classified as REAL or FAKE.

- Combined title and article text used as input  
- Binary labels:  
  - 0 → Fake  
  - 1 → Real  

The dataset is balanced and suitable for supervised learning.

---

## Data Preprocessing

### Text Processing

- Converted text to lowercase  
- Removed URLs and special characters  
- Combined title and article text  
- Tokenized using Keras Tokenizer  
- Applied padding (**max_length = 500**)  

These steps ensured consistent input representation for the model.

---

## Modeling

### Deep Learning Strategy

- Embedding layer for word representation  
- LSTM layer for sequential understanding  
- Dropout for regularization  
- Dense output layer with sigmoid activation  

The model was trained for 5 epochs with validation monitoring.

---

## Results

The model achieved strong performance:

- **Training Accuracy:** ≈ 85%  
- **Validation Accuracy:** ≈ 78%  
- **Test Accuracy:** ≈ 76%  

The confusion matrix shows strong detection of fake news and good generalization performance.

---

## Deployment

The trained model was saved in `.keras` format along with the tokenizer.

The Streamlit application allows users to:

- Enter a news article  
- Get real-time predictions  
- View confidence scores  

---

## Conclusion

This project demonstrates an end-to-end NLP workflow using deep learning.

The LSTM model successfully learned patterns in news text and achieved solid performance on unseen data. Proper preprocessing and structured experimentation were key to achieving reliable results.

---

## How to Run Locally

git clone https://github.com/enesbayraktar61/fake-news-detection-lstm.git

cd fake-news-detection-lstm
pip install -r requirements.txt
streamlit run app.py

---
