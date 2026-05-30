import requests
import json

# URL of the Flask API
url = 'http://127.0.0.1:5000/predict'

# Data to send
data = {
    'text': 'This is a sample news headline.'
}

# Send POST request
response = requests.post(url, json=data)

# Print the response from the Flask app
print(response.json())
