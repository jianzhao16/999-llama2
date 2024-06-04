# -*- coding: utf-8 -*-
import json
import pandas as pd
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from datasets import Dataset, load_dataset
from huggingface_hub import login

# Initialize Hugging Face authentication
hf_auth = "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"
login(token=hf_auth)

# Path to the model

# model_name = "meta-llama/Meta-Llama-3-8B"
#model_name = "/home/tus35240/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3-8B/snapshots/62bd457b6fe961a42a631306577e622c83876cb6"

# model_name = "meta-llama/Llama-2-7B-chat-hf"
model_name = "/home/tus35240/.cache/huggingface/hub/models--meta-llama--Llama-2-7b-chat-hf/snapshots/f5db02db724555f92da89c216ac04704f23d4590"

def process_dataset(file_path, model_name, chunk_size=256, max_length=512):
    # Load the tokenizer directly from the model
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Load custom dataset
    df = pd.read_csv(file_path)

    prompt = df.pop("Prompt")
    comp = df.pop("Completion")
    df["Info"] = prompt + "\n" + comp
    print(df)

    def tokenizing(text, tokenizer, chunk_size, max_length):
        input_ids = []
        attention_masks = []

        for i in range(0, len(text), chunk_size):
            text_chunk = text[i:i+chunk_size]
            encodings = tokenizer(
                text_chunk,
                max_length=max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )

            input_ids.extend(encodings['input_ids'].tolist())
            attention_masks.extend(encodings['attention_mask'].tolist())

        return {'input_ids': input_ids, 'attention_mask': attention_masks}

    print("Tokenizing dataset")
    tokens = tokenizing(list(df["Info"]), tokenizer, chunk_size, max_length)
    tokens_dataset = Dataset.from_dict(tokens)
    print(tokens_dataset)
    print('Tokenizing done')

    return tokens_dataset, tokenizer

# Usage example
file_path = "./medtext_2.csv"
tokens_dataset, tokenizer = process_dataset(file_path, model_name)

# Split the dataset into train and test sets
print("Splitting dataset")
split_dataset = tokens_dataset.train_test_split(test_size=0.2)

# Set up model with 4-bit quantization
from transformers import BitsAndBytesConfig

bb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

falcon_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bb_config
)

# Disable use_cache to be compatible with gradient checkpointing
falcon_model.config.use_cache = False

# Set Training Arguments
training_args = TrainingArguments(
    output_dir='./results',
    overwrite_output_dir=True,
    num_train_epochs=1,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
)

# Initialize the Trainer
trainer = Trainer(
    model=falcon_model,
    args=training_args,
    train_dataset=split_dataset["train"],
    eval_dataset=split_dataset["test"],
    data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
)

# Train the Model
trainer.train()

# Evaluate the Model
results = trainer.evaluate()
print(results)

# Save the Model and Tokenizer
falcon_model.save_pretrained('savemodel')
tokenizer.save_pretrained('savetokenizer')
