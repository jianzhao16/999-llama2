# -*- coding: utf-8 -*-
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import argparse
import os
import torch
from fastapi import FastAPI
from transformers import AutoTokenizer, LlamaForCausalLM

from fastapi import FastAPI, Request
from transformers import AutoTokenizer, LlamaForCausalLM
import torch
from pydantic import BaseModel
from peft import PeftModel, LoraConfig

from transformers import LlamaForCausalLM, AutoTokenizer, FalconForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_int8_training
from huggingface_hub import login
from dotenv import load_dotenv


print('init')
torch.cuda.empty_cache()

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
max_length_set = 256

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def run_ddp(rank, world_size):
    setup(rank, world_size)

    app = FastAPI()

    print('init')
    torch.cuda.empty_cache()

    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
    max_length_set = 256

    # Load environment variables
    print('load env')
    load_dotenv()
    hf_auth = os.getenv("HF_wr")

    # Login to Hugging Face
    print('login ....')
    login(token=hf_auth, add_to_git_credential=True)

    model = './Base-Llama-2-7b-chat-hf'
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
        use_cache=False
    )
    print("model load is done")

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
    model_pretrained_weights = "./myllama2model-ourdata-v2"
    peft_model = PeftModel.from_pretrained(peft_model, model_pretrained_weights)

    # Move model to the appropriate device and wrap with DDP
    device = torch.device(f'cuda:{rank}')
    peft_model.to(device)
    peft_model = DDP(peft_model, device_ids=[rank])

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
        inputs = tokenizer(request.text, return_tensors="pt").to(device)

        # Generate text using the model
        outputs = peft_model.module.generate(input_ids=inputs.input_ids, max_new_tokens=request.max_new_tokens)

        # Decode the generated tokens to text
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Return the generated text
        return {"generated_text": generated_text}

    if rank == 0:
        import uvicorn
        # Run the FastAPI application with uvicorn
        uvicorn.run(app, host="0.0.0.0", port=8000)

    cleanup()

def main():
    world_size = torch.cuda.device_count()
    mp.spawn(run_ddp, args=(world_size,), nprocs=world_size, join=True)

if __name__ == "__main__":
    main()
