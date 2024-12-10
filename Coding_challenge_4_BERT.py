import re
import os
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset


###############################################################################################
# Data-prep
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

file_paths = ['amazon_cells_labelled.txt', 'imdb_labelled.txt', 'yelp_labelled.txt']
data = load_data(file_paths)

# Check class distribution
print(data['label'].value_counts())

X = data['text'].values
y = data['label'].values

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


###############################################################################################
# Model prep
###############################################################################################


# Load the BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Tokenize and pad sequences
max_length = 250
train_encodings = tokenizer(list(X_train), truncation=True, padding=True, max_length=max_length)
test_encodings = tokenizer(list(X_test), truncation=True, padding=True, max_length=max_length)

# Convert to PyTorch Dataset
train_inputs = torch.tensor(train_encodings['input_ids'])
train_masks = torch.tensor(train_encodings['attention_mask'])
train_labels = torch.tensor(y_train)

test_inputs = torch.tensor(test_encodings['input_ids'])
test_masks = torch.tensor(test_encodings['attention_mask'])
test_labels = torch.tensor(y_test)

train_dataset = TensorDataset(train_inputs, train_masks, train_labels)
test_dataset = TensorDataset(test_inputs, test_masks, test_labels)

# DataLoader for batching
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=32)


# Load BERT model
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

# # Freeze all but the dense layers of BERT
# for param in model.bert.parameters():
#     param.requires_grad = False

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Optimizer
optimizer = AdamW(model.parameters(), lr=1e-5)

# Learning rate scheduler
lr_scheduler = ReduceLROnPlateau(optimizer, factor=0.5, patience=2, min_lr=1e-6, verbose=True)


###############################################################################################
# Training and validation execution
###############################################################################################


# Training Loop
def train_epoch(model, data_loader, optimizer, device):
    model.train()
    total_loss = 0
    correct_predictions = 0
    for batch in data_loader:
        input_ids, attention_mask, labels = [b.to(device) for b in batch]
        
        optimizer.zero_grad()
        
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        logits = outputs.logits
        
        total_loss += loss.item()
        
        # Convert logits to predictions
        preds = torch.argmax(logits, dim=1)
        correct_predictions += torch.sum(preds == labels).item()
        
        loss.backward()
        optimizer.step()
    
    return total_loss / len(data_loader), correct_predictions / len(data_loader.dataset)

# Validation Loop
def eval_epoch(model, data_loader, device):
    model.eval()
    total_loss = 0
    correct_predictions = 0
    with torch.no_grad():
        for batch in data_loader:
            input_ids, attention_mask, labels = [b.to(device) for b in batch]
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            logits = outputs.logits
            
            total_loss += loss.item()
            
            # Convert logits to predictions
            preds = torch.argmax(logits, dim=1)
            correct_predictions += torch.sum(preds == labels).item()
    
    return total_loss / len(data_loader), correct_predictions / len(data_loader.dataset)


# Training and Evaluation Loop
epochs = 4
train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []

for epoch in range(epochs):
    print(f"Epoch {epoch + 1}/{epochs}")
    
    train_loss, train_accuracy = train_epoch(model, train_dataloader, optimizer, device)
    val_loss, val_accuracy = eval_epoch(model, test_dataloader, device)
    
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    train_accuracies.append(train_accuracy)
    val_accuracies.append(val_accuracy)
    
    print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")
    print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")
    
    # Update learning rate
    lr_scheduler.step(val_loss)


###############################################################################################
# Plotting and model save
###############################################################################################


# Plot training history
plt.figure(figsize=(12, 4))

# Plot accuracy
plt.subplot(1, 2, 1)
plt.plot(train_accuracies, label='Training Accuracy')
plt.plot(val_accuracies, label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Plot loss
plt.subplot(1, 2, 2)
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()

# Save the plot
plt.savefig('Figure_BERT_PyTorch.png', dpi=300)  # Save as a high-resolution image

plt.show()

# Save the Model and tokenizer
model.save_pretrained('BERT_PyTorch')
tokenizer.save_pretrained('tokenizer')
