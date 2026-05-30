from flask import Flask, request, jsonify
import pickle

# Initialize Flask app
app = Flask(__name__)

# Load saved model and vectorizer
with open("fake_news_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

with open("tfidf_vectorizer.pkl", "rb") as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

@app.route('/')
def home():
    return "Fake News Detection API is running!"

# Define prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    # Get text from the request
    data = request.get_json(force=True)
    text = data['text']
    
    # Transform text and predict
    text = vectorizer.transform([text])
    prediction = model.predict(text)
    
    # Return the result
    return jsonify({'prediction': 'Real' if prediction[0] == 1 else 'Fake'})

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
