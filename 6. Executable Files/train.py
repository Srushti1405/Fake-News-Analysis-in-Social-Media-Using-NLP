import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import csv

# Load the dataset with specific quoting and encoding settings
try:
    data = pd.read_csv(
        'news (3).csv', 
        error_bad_lines=False,  # Skip problematic rows
        quoting=csv.QUOTE_ALL,  # Adjust this as needed
        encoding='ISO-8859-1'   # Adjust encoding if needed
    )
except pd.errors.ParserError:
    raise ValueError("Failed to parse CSV file. Please check the file format.")

# Strip any whitespace from column names
data.columns = data.columns.str.strip()

# Check unique values in the 'label' column before mapping
print("Unique values in 'label' column before mapping:", data['label'].unique())

# Clean the 'label' column by filtering out any unexpected values
valid_labels = ['FAKE', 'REAL']  # Use uppercase values as in the dataset
data = data[data['label'].isin(valid_labels)]

# Drop rows with missing or invalid 'label' and 'text' values
data = data.dropna(subset=['label', 'text'])

# Check unique values in 'label' column after cleaning
print("Unique values in 'label' column after cleaning:", data['label'].unique())

# Ensure that the 'label' column contains valid values ('FAKE' or 'REAL')
if data['label'].isin(valid_labels).all() == False:
    raise ValueError("The 'label' column must only contain 'FAKE' or 'REAL' values.")

# Map 'label' values to numeric (1 for fake news, 0 for real news)
data['label'] = data['label'].map({'FAKE': 1, 'REAL': 0})

# Check unique values in 'label' column after mapping
print("Unique values in 'label' column after mapping:", data['label'].unique())

# Check for any NaN or infinite values after mapping
if data['label'].isnull().any() or (data['label'] == np.inf).any():
    raise ValueError("The 'label' column still contains invalid values after preprocessing.")

# Separate text and labels
texts = data['text']
labels = data['label'].astype(int)

# Tokenize and pad sequences
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(texts)
X = tokenizer.texts_to_sequences(texts)
X = pad_sequences(X, maxlen=100)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)

# Build the LSTM model
model = Sequential([
    Embedding(input_dim=5000, output_dim=128, input_length=100),
    LSTM(64),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Fit the model
model.fit(X_train, y_train, epochs=5, batch_size=32, validation_data=(X_test, y_test))

# Save the model and tokenizer
model.save('model/fake_news_model.h5')
with open('model/tokenizer.pkl', 'wb') as f:
    pickle.dump(tokenizer, f)

print("Model and tokenizer have been saved successfully.")
