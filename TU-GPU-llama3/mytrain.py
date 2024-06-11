import argparse
from collections import deque
import pandas as pd
import torch
from transformers import AutoTokenizer

from llama import Llama


# Data loading and preprocessing
def load_and_preprocess_data(file_path, tokenizer_name):
    data = pd.read_csv(file_path)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    if 'text' in data.columns and 'label' in data.columns:
        tokenized_data = tokenizer(data['text'].tolist(), padding="max_length", truncation=True, max_length=512)
        labels = data['label'].tolist()
        return tokenized_data, labels, 'classification'
    elif 'input_text' in data.columns and 'target_text' in data.columns:
        tokenized_inputs = tokenizer(data['input_text'].tolist(), padding="max_length", truncation=True, max_length=512)
        tokenized_targets = tokenizer(data['target_text'].tolist(), padding="max_length", truncation=True,
                                      max_length=512)
        return tokenized_inputs, tokenized_targets, 'generation'
    else:
        raise ValueError("CSV format not recognized. Please include the correct columns.")


# Model training function
def train_model(ckpt_dir, tokenizer_path, dataset_path, max_seq_len, max_batch_size):
    print("Loading and preprocessing dataset...")
    processed_data, labels_or_targets, task_type = load_and_preprocess_data(dataset_path, tokenizer_path)

    print("Initializing the model for training...")
    model = Llama.build(ckpt_dir=ckpt_dir, tokenizer_path=tokenizer_path, max_seq_len=max_seq_len,
                        max_batch_size=max_batch_size)

    print("Starting training...")
    if task_type == 'classification':
        # Placeholder for classification training code
        # model.train_classification(processed_data, labels_or_targets)
        pass
    elif task_type == 'generation':
        # Placeholder for generation training code
        # model.train_generation(processed_data, labels_or_targets)
        pass
    print("Training completed.")
    model.save(ckpt_dir)
    print(f"Model saved in {ckpt_dir}")


# Chatbot running function using trained model
def run_chatbot(ckpt_dir, tokenizer_path, task_type):
    print("Initializing LLaMA 3 8B model...")
    generator = Llama.build(ckpt_dir=ckpt_dir, tokenizer_path=tokenizer_path, max_seq_len=256, max_batch_size=4)

    try:
        while True:
            current_input = input("Enter your instruction (type 'exit' to finish): ")
            if current_input.lower() == "exit":
                break
            if current_input == "":
                print('Empty instruction, please try again.')
                continue

            tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
            inputs = tokenizer(current_input, return_tensors="pt", padding="max_length", truncation=True,
                               max_length=512)

            if task_type == 'classification':
                outputs = generator.predict(inputs)
                print("\nClassification Label:", outputs)
            elif task_type == 'generation':
                outputs = generator.generate(**inputs)
                generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                print("\nGenerated Text:", generated_text)

    except KeyboardInterrupt:
        print("Interrupted by user.")

    print("Session Ended.")


# Main function setup for command line arguments
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the chatbot with model training capabilities.")
    parser.add_argument("--ckpt_dir", type=str, required=True, help="Checkpoint directory for the model.")
    parser.add_argument("--tokenizer_path", type=str, required=True, help="Path to the tokenizer.")
    parser.add_argument("--dataset_path", type=str, help="Path to the custom training dataset.")
    parser.add_argument("--task_type", type=str, choices=['classification', 'generation'], required=True,
                        help="Type of task to perform.")
    parser.add_argument("--train", action='store_true', help="Flag to train the model.")
    args = parser.parse_args()

    max_seq_len = 256  # You can parameterize this if needed
    max_batch_size = 4  # You can

    if args.train and args.dataset_path:
        print("Training the model...")
        train_model(args.ckpt_dir, args.tokenizer_path, args.dataset_path, max_seq_len, max_batch_size)
    else:
        print("Load Pretained Model, Running the chatbot...")
        run_chatbot(args.ckpt_dir, args.tokenizer_path, args.task_type)

