import re
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
from collections import Counter
import nltk


# Download Punkt tokenizer (if not already downloaded)
nltk.download('punkt')
from nltk.tokenize import sent_tokenize


# Load the pretrained Keras model and tokenizer
model = load_model("challenge42.keras")
with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)


# Function to preprocess text for the Keras model
def preprocess_text(text, tokenizer, max_length=100):
    # Split text into sentences using nltk's sent_tokenize
    sentences = sent_tokenize(text)
    
    # Tokenize and pad sentences
    sequences = tokenizer.texts_to_sequences(sentences)
    padded_sequences = pad_sequences(sequences, maxlen=max_length, padding='post', truncating='post')

    return sentences, padded_sequences


# Function to classify sentences using the Keras model
def classify_sentences(model, sentences, padded_sequences):
    predictions = model.predict(padded_sequences)
    sentiments = ['Positive' if p > 0.5 else 'Negative' for p in predictions]
    return list(zip(sentences, sentiments, predictions.flatten()))


# Load the book text
def load_text(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()


# Plot Sentiment Distribution
def plot_sentiment_distribution(results, title, output_file):
    sentiments = [result[1] for result in results]  # Extract sentiments
    sentiment_counts = Counter(sentiments)

    labels = list(sentiment_counts.keys())
    values = list(sentiment_counts.values())

    plt.figure(figsize=(8, 6))
    plt.bar(labels, values, color=['skyblue', 'salmon'])
    plt.title(title)
    plt.xlabel("Sentiment")
    plt.ylabel("Count")
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    for i, value in enumerate(values):
        plt.text(i, value + 1, str(value), ha='center', fontsize=12)

    plt.tight_layout()
    plt.savefig(output_file)
    plt.show()




# Main script execution
if __name__ == "__main__":
    book_text_path = "notes_from_the_underground.txt"
    book_text = load_text(book_text_path)

    sentences, padded_sequences = preprocess_text(book_text, tokenizer)
    results = classify_sentences(model, sentences, padded_sequences)

    # Save results to file
    output_file = "sentiment_results_my_model.txt"
    with open(output_file, 'w', encoding='utf-8') as f:
        for sentence, sentiment, score in results:
            f.write(f"Sentence: {sentence}\nSentiment: {sentiment}\nScore: {score:.4f}\n\n")
    print(f"Keras classification complete. Results saved to {output_file}.")

    # Plot sentiment distribution
    plot_sentiment_distribution(results, "Keras Sentiment Distribution", "sentiment_distribution_keras.png")
