# train_llama3.py

import pandas as pd
import torch
from transformers import LlamaTokenizer, LlamaForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split

# Load the dataset
df = pd.read_csv('your_dataset.csv')  # Replace 'your_dataset.csv' with your actual dataset file path

# Split the dataset into train and test sets
train_texts, test_texts, train_labels, test_labels = train_test_split(df['text'].tolist(), df['label'].tolist(), test_size=0.2)

# Initialize the tokenizer
tokenizer = LlamaTokenizer.from_pretrained('path_to_llama3_tokenizer')  # Replace with your tokenizer path

# Tokenize the data
train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=256)
test_encodings = tokenizer(test_texts, truncation=True, padding=True, max_length=256)

class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = Dataset(train_encodings, train_labels)
test_dataset = Dataset(test_encodings, test_labels)

# Initialize the model
model = LlamaForSequenceClassification.from_pretrained('path_to_llama3_model')  # Replace with your model path

# Set up training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)

# Train the model
trainer.train()

# Save the model
model.save_pretrained('path_to_save_model')  # Replace with your desired save path
tokenizer.save_pretrained('path_to_save_tokenizer')  # Replace with your desired save path

# Function to make predictions
def predict(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=256)
    outputs = model(**inputs)
    predictions = torch.argmax(outputs.logits, dim=-1)
    return predictions.item()

# Example usage
text = "Your input text here"
prediction = predict(text)
print(f'Prediction: {prediction}')
