# -*- coding: utf-8 -*-
import os
import pandas as pd
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForSeq2Seq
import torch
import os


def check_gpu_memory():
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        print(f"Number of GPUs available: {num_gpus}")
        for gpu_id in range(num_gpus):
            torch.cuda.set_device(gpu_id)
            total_memory = torch.cuda.get_device_properties(gpu_id).total_memory
            free_memory = total_memory - torch.cuda.memory_allocated(gpu_id)
            total_memory_gb = total_memory / (1024 ** 3)
            free_memory_gb = free_memory / (1024 ** 3)
            print(f"GPU {gpu_id}: Total Memory: {total_memory_gb:.2f} GB, Free Memory: {free_memory_gb:.2f} GB")
    else:
        print("CUDA is not available. Check your installation or GPU availability.")

#os.environ["CUDA_VISIBLE_DEVICES"] = '1'
# Set environment variable for memory management
#os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Clear GPU memory
torch.cuda.empty_cache()

check_gpu_memory()

# Load dataset from CSV file
df = pd.read_csv('ourdata.csv')

# Combine "Prompt" and "Completion" columns into "Info" column
df["Info"] = df["input_text"] + "\n" + df["target_text"]

# Convert DataFrame to Hugging Face Dataset
dataset = Dataset.from_pandas(df)

# Path to the local directory where the model and tokenizer files are stored
model_path = 'meta-llama/Llama-2-7b-hf'

# Initialize tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Add padding token if it doesn't exist
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

# Load the model and resize token embeddings
model = AutoModelForCausalLM.from_pretrained(model_path)
model.resize_token_embeddings(len(tokenizer))

# Tokenize dataset using Hugging Face's method
def tokenize_function(examples):
    return tokenizer(examples["Info"], padding="max_length", truncation=True, max_length=512)

tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=["Info", "input_text", "target_text"])

# Split the dataset into training and validation sets
split_dataset = tokenized_datasets.train_test_split(test_size=0.2)
print('split dataset is done')


print('begin training: set arguments')
# Set up training arguments
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    save_total_limit=2,
    save_steps=500,
    eval_strategy="steps",
    eval_steps=500,
    load_best_model_at_end=True,
)

# Initialize the Data Collator
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

# Initialize the Trainer
class NoGradTrainer(Trainer):
    def evaluation_step(self, model, inputs):
        """Overriding the evaluation_step method to use torch.no_grad()"""
        model.eval()
        with torch.no_grad():
            outputs = model(**inputs)
        model.train()
        return outputs

trainer = NoGradTrainer(
    model=model,
    args=training_args,
    train_dataset=split_dataset['train'],
    eval_dataset=split_dataset['test'],
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# Clear GPU memory before training
torch.cuda.empty_cache()

# Train the model
trainer.train()

# Evaluate the model
trainer.evaluate()

# Save the model and tokenizer
model.save_pretrained("path_to_save_model")
tokenizer.save_pretrained("path_to_save_tokenizer")





