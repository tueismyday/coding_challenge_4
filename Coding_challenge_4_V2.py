import tensorflow as tf
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional, Layer, Input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import re
import nltk
from nltk.corpus import stopwords
from tensorflow.keras.layers import Attention
from tensorflow.keras.layers import LayerNormalization
import matplotlib.pyplot as plt


###############################################################################################
# Data load and cleaning definition
###############################################################################################

# Ensure stopwords are downloaded
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Function to clean text
def clean_text(text):
    text = text.lower()                                         # Convert to lowercase
    text = re.sub(r'\d+', '', text)                             # Remove numbers
    text = re.sub(r'[^\w\s]', '', text)                         # Remove punctuation
    text = re.sub(r'\s+', ' ', text).strip()                    # Remove extra whitespace
    words = text.split()
    words = [word for word in words if word not in stop_words]  # Remove stopwords
    return ' '.join(words)

# Load and Preprocess Data
def load_data(file_paths):
    data = []
    for file_path in file_paths:
        with open(file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
            for line in lines:
                text, label = line.rsplit('\t', 1)
                text = clean_text(text.strip())  #### Apply text cleaning
                data.append((text, int(label.strip())))
    return pd.DataFrame(data, columns=['text', 'label'])


###############################################################################################
# Text preprocessing
###############################################################################################


file_paths = ['amazon_cells_labelled.txt', 'imdb_labelled.txt', 'yelp_labelled.txt']
data = load_data(file_paths)

# Check class distribution
print(data['label'].value_counts())


X = data['text'].values
y = data['label'].values

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Tokenization and Padding
vocab_size = 100000
max_length = 250
tokenizer = Tokenizer(num_words=vocab_size, oov_token='<OOV>')
tokenizer.fit_on_texts(X_train)

X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)

X_train_padded = pad_sequences(X_train_seq, maxlen=max_length, padding='post', truncating='post')
X_test_padded = pad_sequences(X_test_seq, maxlen=max_length, padding='post', truncating='post')


###############################################################################################
# Build the LSTM Model
###############################################################################################


embedding_dim = 256

model = Sequential([
    Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_length),
    Bidirectional(LSTM(256, return_sequences=False)),
    Dropout(0.5),
    Dense(128, activation='relu', kernel_regularizer='l2'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')  # Sigmoid for binary classification
])

lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6)

# Compile the Model
optimizer = Adam(learning_rate=1e-4)
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

# Train the Model
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

history = model.fit(
    X_train_padded, y_train,
    validation_data=(X_test_padded, y_test),
    epochs=50,
    batch_size=100,
    callbacks=[early_stopping, lr_scheduler],
)

# Evaluate the Model
loss, accuracy = model.evaluate(X_test_padded, y_test)
print(f"Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}")

# Model Summary
model.summary()

###############################################################################################
# Plot training history
###############################################################################################


plt.figure(figsize=(12, 4))

# Plot accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Plot loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()

# Save the plot
plt.savefig('Figure_V2.png', dpi=300)  # Save as a high-resolution image

plt.show()


###############################################################################################
# Save the Model and tokenizer
###############################################################################################

model.save('challenge42.keras')

import pickle
with open('tokenizer.pkl', 'wb') as f:
    pickle.dump(tokenizer, f)
