# Fake News Detection API

A machine learning-based Fake News Detection system that classifies news articles as **Real** or **Fake** using **TF-IDF feature extraction** and **Logistic Regression**. The trained model is deployed through a **Flask REST API**, allowing users to submit news text and receive predictions in real time.

---

## Overview

Fake news and misinformation have become major challenges in the digital era. This project demonstrates a simple Natural Language Processing (NLP) pipeline that learns textual patterns from news articles and predicts whether a given news item is authentic or misleading.

The project covers the complete workflow:

* Data preprocessing
* Feature extraction using TF-IDF
* Model training using Logistic Regression
* Model serialization using Pickle
* REST API deployment with Flask
* API testing using Python Requests

---

## Features

* Binary classification of news articles
* TF-IDF text vectorization
* Logistic Regression classifier
* Model persistence using Pickle
* Flask-based REST API
* Simple API testing script
* Beginner-friendly implementation

---

## Project Structure

```text
fake-news-detection-api/
│
├── app.py                    # Flask API application
├── trainFakeNewsModel.py     # Model training script
├── testApi.py                # API testing script
├── fake_news_model.pkl       # Trained Logistic Regression model
├── tfidf_vectorizer.pkl      # Saved TF-IDF vectorizer
└── README.md
```

---

## Machine Learning Pipeline

### 1. Data Preparation

* Load fake and real news datasets
* Merge datasets into a single dataframe
* Assign labels for classification
* Shuffle records to improve training quality

### 2. Text Processing

News content is converted into numerical representations using:

* TF-IDF Vectorization
* English stop-word removal
* Vocabulary filtering

### 3. Model Training

The project uses:

**Logistic Regression**

for binary classification of news articles.

### 4. Model Deployment

The trained model and vectorizer are saved using Pickle and loaded into a Flask application for inference.

---

## Technologies Used

* Python
* Scikit-learn
* Pandas
* NumPy
* Flask
* Pickle
* Requests

---

## Installation

### Clone the Repository

```bash
git clone https://github.com/your-username/fake-news-detection-api.git
cd fake-news-detection-api
```

### Install Dependencies

```bash
pip install pandas numpy scikit-learn flask requests
```

---

## Training the Model

Run:

```bash
python trainFakeNewsModel.py
```

This will:

* Train the Logistic Regression model
* Generate TF-IDF features
* Save the trained model
* Save the vectorizer

Generated files:

```text
fake_news_model.pkl
tfidf_vectorizer.pkl
```

---

## Running the API

Start the Flask server:

```bash
python app.py
```

Server:

```text
http://127.0.0.1:5000
```

---

## API Endpoint

### POST /predict

#### Request

```json
{
    "text": "Breaking news article content goes here"
}
```

#### Response

```json
{
    "prediction": "Real"
}
```

or

```json
{
    "prediction": "Fake"
}
```

---

## Testing the API

Run:

```bash
python testApi.py
```

The script sends a sample request to the Flask server and prints the prediction response.


---

## Learning Outcomes

This project demonstrates:

* Text preprocessing techniques
* Feature engineering using TF-IDF
* Binary text classification
* Machine learning model deployment
* REST API development using Flask
* End-to-end NLP application development

---

## Future Improvements

* Support for complete news articles and URLs
* Advanced NLP preprocessing
* Deep Learning models (LSTM, BERT)
* Web-based user interface
* Model performance visualization
* Docker deployment

---

**Manoj Kumar Sunkara**

Artificial Intelligence & Machine Learning

---

