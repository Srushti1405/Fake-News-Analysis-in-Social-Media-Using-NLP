from flask import Flask, request, render_template
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

app = Flask(__name__)

# Load the trained model and tokenizer
model = load_model('model/fake_news_model.h5')  # Make sure the model path is correct
with open('model/tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

def predict_fake_news(text):
    # Tokenize and pad the input text
    sequences = tokenizer.texts_to_sequences([text])
    padded_sequence = pad_sequences(sequences, maxlen=100)
    
    # Get the prediction from the model
    prediction = model.predict(padded_sequence)[0][0]  # prediction is a scalar value
    
    print(f"Prediction value: {prediction}")  # Debugging the prediction
    
    # If prediction > 0.5, classify it as Fake News, otherwise Real News
    return 'Fake News' if prediction > 0.5 else 'Real News'

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        news_text = request.form['news']
        print(f"Received news text: {news_text}")  # Debugging: Check if news text is received
        result = predict_fake_news(news_text)
        print(f"Prediction result: {result}")  # Debugging: Check the result before returning to template
        return render_template('index.html', prediction=result, news=news_text)
    
    # Render the form when it's a GET request
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
