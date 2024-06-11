# -*- coding: utf-8 -*-
from fastapi import FastAPI, Request
from transformers import AutoTokenizer, LlamaForCausalLM
import torch
from pydantic import BaseModel
from peft import PeftModel, LoraConfig

from transformers import LlamaForCausalLM, AutoTokenizer, FalconForCausalLM, BitsAndBytesConfig

from peft import LoraConfig, get_peft_model, prepare_model_for_int8_training

import torch

from huggingface_hub import login

import os

from dotenv import load_dotenv

app = FastAPI()

print('init')
torch.cuda.empty_cache()

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
max_length_set = 256

# Initialize Hugging Face authentication
# custom
#hf_auth = "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"

# write
print('load env')
# Initialize Hugging Face authentication
# Load .env file
load_dotenv()
# Retrieve API key
hf_auth = os.getenv("HF_wr")
#hf_auth = "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"
#login(token=hf_auth)
print('login ....')
login(token=hf_auth,add_to_git_credential=True)


#model = 'meta-llama/Llama-2-7b-chat-hf'
model = './Base-Llama-3-8B-Instruct'
print(model)


tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

print('tokenizer load')

bb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

print("bb_config load is done")

base_model = LlamaForCausalLM.from_pretrained(
    model,
    quantization_config=bb_config,
    low_cpu_mem_usage=True,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    #trust_remote_code=True,
    use_cache=False
)
print("model load is done")

#base_model.to("cuda" if torch.cuda.is_available() else "cpu")

print("Preparing gradient model")
base_model.gradient_checkpointing_enable()
base_model = prepare_model_for_int8_training(base_model)
print("Preparing Lora model parameters")


# Define and apply PEFT configuration (LoRA)
lora_config = LoraConfig(
    r=4,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

# Wrap the base model with PEFT model
peft_model = PeftModel(base_model, lora_config)


print('load 1 perf model weights')
# load 1
model_pretrained_weights = "./myllama3model-ourdata-v2"
peft_model = PeftModel.from_pretrained(peft_model, model_pretrained_weights)
#peft_model = peft_model.merge_and_unload()

print('load 2 peft model weights')
# Load the fine-tuned weights
#model_pretrained_weights = "./myllama3model"
#peft_model.load_state_dict(torch.load(f"{model_pretrained_weights}/pytorch_model.bin"))

# Move model to GPU if available
#peft_model.to("cuda" if torch.cuda.is_available() else "cpu")


# Define request model for the API
class TextRequest(BaseModel):
    text: str
    max_new_tokens: int = 128


@app.post("/generate-text")
async def generate_text(request: TextRequest):
    """
    Endpoint to generate text based on input prompt.
    """
    # Tokenize input text
    inputs = tokenizer(request.text, return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")

    # Generate text using the model
    outputs = peft_model.generate(input_ids=inputs.input_ids, max_new_tokens=request.max_new_tokens)

    # Decode the generated tokens to text
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Return the generated text
    return {"generated_text": generated_text}


if __name__ == "__main__":
    import uvicorn

    # Run the FastAPI application with uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
