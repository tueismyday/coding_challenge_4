import re
import torch
import numpy as np
import matplotlib.pyplot as plt
import nltk
from nltk.tokenize import sent_tokenize
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader, TensorDataset
from collections import Counter

model_path = 'BERT_PyTorch'  
tokenizer_path = 'tokenizer' 

# Download Punkt tokenizer (if not already downloaded)
nltk.download('punkt')

# Check if a GPU is available and use it, otherwise fallback to CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the fine-tuned BERT model and tokenizer (PyTorch)
model = BertForSequenceClassification.from_pretrained(model_path)
tokenizer = BertTokenizer.from_pretrained(tokenizer_path)

# Make sure the model is in evaluation mode
model.eval()

# Function to preprocess text for BERT model
def preprocess_text(text, tokenizer, max_length=250):
    # Split text into sentences using nltk's sent_tokenize
    sentences = sent_tokenize(text)
    
    # Tokenize and pad sentences
    inputs = tokenizer(sentences, return_tensors='pt', truncation=True, padding=True, max_length=max_length, add_special_tokens=True)
    return sentences, inputs

# Function to classify sentences using the fine-tuned BERT model
def classify_sentences(model, inputs):
    with torch.no_grad():
        # Run the model to get predictions
        outputs = model(**inputs)
        logits = outputs.logits  # Get raw logits from the model
        
        # Apply softmax to get probabilities
        probs = torch.nn.functional.softmax(logits, dim=-1)
        
        # Convert probabilities to predicted labels
        predictions = torch.argmax(probs, dim=-1).cpu().numpy()
        scores = probs[:, 1].cpu().numpy()  # Probabilities for the positive class
    return predictions, scores

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

    # Preprocess the text and get the tokenized inputs for BERT
    sentences, inputs = preprocess_text(book_text, tokenizer)

    # Classify the sentences and get the predictions
    predictions, scores = classify_sentences(model, inputs)

    # Map predictions to sentiments ('Positive' / 'Negative')
    sentiments = ['Positive' if p == 1 else 'Negative' for p in predictions]

    # Combine sentences, sentiments, and scores
    results = list(zip(sentences, sentiments, scores))

    # Save results to file
    output_file = "sentiment_results_bert.txt"
    with open(output_file, 'w', encoding='utf-8') as f:
        for sentence, sentiment in results:
            f.write(f"Sentence: {sentence}\nSentiment: {sentiment}\n\n")
    print(f"BERT classification complete. Results saved to {output_file}.")

    # Plot sentiment distribution
    plot_sentiment_distribution(results, "BERT Sentiment Distribution", "sentiment_distribution_bert_fine_tuned.png")
