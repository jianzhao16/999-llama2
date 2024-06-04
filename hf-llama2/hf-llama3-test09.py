import transformers
import torch

# -*- coding: utf-8 -*-

# export PYTHONIOENCODING=utf-8
# python your_script.py
import torch
from transformers import LlamaForCausalLM, AutoTokenizer, FalconForCausalLM, BitsAndBytesConfig, TrainingArguments, \
    Trainer, DataCollatorForLanguageModeling
import transformers
from peft import LoraConfig, get_peft_model, prepare_model_for_int8_training
from datasets import load_dataset, Dataset
import pandas as pd
import torch

from huggingface_hub import login

import os

torch.cuda.empty_cache()

#export 'PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'


# Initialize Hugging Face authentication
hf_auth = "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"
login(token=hf_auth)

# Path to the model
model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
# model_name = "/home/tus35240/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3-8B/snapshots/62bd457b6fe961a42a631306577e622c83876cb6"

# model_name = "meta-llama/Llama-2-7B-chat-hf"


# model = "/home/tus35240/.cache/huggingface/hub/models--meta-llama--Llama-2-7b-chat-hf/snapshots/f5db02db724555f92da89c216ac04704f23d4590"
# model = "ybelkada/falcon-7b-sharded-bf16"

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

bb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

print("bb_config load is done")

falcon_model = LlamaForCausalLM.from_pretrained(
    model_name,
    quantization_config=bb_config,
    use_cache=False
)
print("model load is done")



text_pad1 = "Question: What is the national bird of the United States? \n Answer: "

# Tokenize the input
inputs = tokenizer(text_pad1, return_tensors="pt", padding=True).to("cuda:0")

# Set the attention mask
attention_mask = inputs["attention_mask"]

# Generate output with the specified attention mask and pad token id
outputs = falcon_model.generate(
    inputs["input_ids"],
    attention_mask=attention_mask,
    max_new_tokens=100,
    pad_token_id=tokenizer.eos_token_id
)

# Decode the generated output
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(generated_text)

print('test is done')


'''
pipeline = transformers.pipeline(
    "text-generation",
    model=falcon_model,
    model_kwargs={"torch_dtype": torch.bfloat16},
    tokenizer=tokenizer,
    device="cuda",
)

output = pipeline("Once upon a time,")
print(output)
'''
