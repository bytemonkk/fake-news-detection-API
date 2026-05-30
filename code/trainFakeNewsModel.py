import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# load The datasets dobby..!
fake_df = pd.read_csv("E:/FAKE NEWS DETECTION/data/Fake.csv")
real_df = pd.read_csv("E:/FAKE NEWS DETECTION/data/True.csv")

# cleaned One..!
'''print(fake_df.isnull().sum())
print(real_df.isnull().sum())'''

fake_df['label'] = 1
real_df['label'] = 0

# combine The datasets dobby..!
data = pd.concat([fake_df, real_df], ignore_index=True)

# now shuffle The dataset dobby..!
#bcoz cross-validation comes into The picture..!
data = data.sample(frac=1, random_state=42).reset_index(drop=True)

data['combined_text'] = data['title'] + " " + data['text'] + " " + data['subject']

x = data['combined_text']
y = data['label']

# split The dataset dobby..!

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# convert text To numeric data dobby..!

from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)

x_train = vectorizer.fit_transform(x_train)
x_test = vectorizer.transform(x_test)

# let's Train The model dobby..!

from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(x_train, y_train)

# Test The model dobby..!

y_pred = model.predict(x_test)
print(np.concatenate([np.array(y_pred).reshape(-1, 1), np.array(y_test).reshape(-1, 1)] ,1))

# Evaluate The model dobby..!

from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
# acc_s = accuracy_score(y_test, y_pred)

print(cm)

# Evaluate on training data
y_train_pred = model.predict(x_train)
train_accuracy = accuracy_score(y_train, y_train_pred)

# Evaluate on test data
y_test_pred = model.predict(x_test)
test_accuracy = accuracy_score(y_test, y_test_pred)

print(f"Train Accuracy: {train_accuracy:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")


# let's save The model for later use dobby..!

import pickle

# save model
with open("fake_news_model.pkl", "wb") as f:
    pickle.dump(model, f)

# save vectorizer
with open("tfidf_vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)


# Load model and vectorizer

with open("fake_news_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

with open("tfidf_vectorizer.pkl", "rb") as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

# New news headline for prediction
new_text = ["The president announces new global initiatives"]

# Transform the new text using the vectorizer
new_text = vectorizer.transform(new_text)

# Make the prediction
prediction = model.predict(new_text)

# Output prediction
print(f"Prediction: {'Fake' if prediction[0] == 0 else 'Real'}")

    

