import os
from threading import Thread

from transformers import AutoTokenizer, TextIteratorStreamer, AutoModelForCausalLM, LlamaForCausalLM
import transformers
import torch
from huggingface_hub import login

# !pip install transformers==4.31.0
# !pip install accelerate==0.21.0
# !pip install bitsandbytes==0.40.2
# !pip install peft==0.4.0
# !pip install datasets
# !pip install einops
# !pip install torch==2.0.1


# -*- coding: utf-8 -*-
from datetime import time

# export PYTHONIOENCODING=utf-8
# python your_script.py
from transformers import LlamaForCausalLM, AutoTokenizer, FalconForCausalLM, BitsAndBytesConfig, TrainingArguments, \
    Trainer, DataCollatorForLanguageModeling

import torch

from huggingface_hub import login

import os

from dotenv import load_dotenv


print('init')
torch.cuda.empty_cache()

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
max_length_set = 256

print('load env')
# Initialize Hugging Face authentication
# Load .env file
load_dotenv()
# Retrieve API key
hf_auth = os.getenv("HF_wr")

print('login ....')
login(token=hf_auth,add_to_git_credential=True)

# Path to the model
#model = "meta-llama/Meta-Llama-3-8B-Instruct"
model = 'Base-Llama-2-7b-chat-hf'
max_new_tokens=128
#model = 'meta-llama/Llama-2-7b-chat-hf'
print(model)
# model_name = "/home/tus35240/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3-8B/snapshots/62bd457b6fe961a42a631306577e622c83876cb6"

# model_name = "meta-llama/Llama-2-7B-chat-hf"


# model = "/home/tus35240/.cache/huggingface/hub/models--meta-llama--Llama-2-7b-chat-hf/snapshots/f5db02db724555f92da89c216ac04704f23d4590"
# model = "ybelkada/falcon-7b-sharded-bf16"

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

pre_model = LlamaForCausalLM.from_pretrained(
    model,
    quantization_config=bb_config,
    low_cpu_mem_usage=True,
    use_cache=False
)
print("model load is done")


print('*****************')

print('test input')
inputs = tokenizer(["Explain Random Forest in 5 key points and maximum 100 words."], return_tensors="pt")
inputs["input_ids"] = inputs["input_ids"].to("cuda:0")
print(inputs)
streamer = TextIteratorStreamer(tokenizer,skip_prompt=True)
print('streamer is as follows')
print(streamer)

print('test generation')
generation_kwargs = dict(inputs, streamer=streamer, max_new_tokens=1500)
print('generation_kwargs is as follows')
print(generation_kwargs)

print('Begin Run the model to generate text')
thread = Thread(target=pre_model.generate, kwargs=generation_kwargs)
print('thread is running')

thread.start()
print('thread is started')
generation_text= ""


print('streamer is as follows')
print(streamer)
for new_text in streamer:
    print(new_text,end='')
    #print(tokenizer.decode(new_text))





'''
from transformers import StoppingCriteria, StoppingCriteriaList

generated_tokens = []
# define custom stopping criteria object
class StopOnTokens(StoppingCriteria):
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        #for stop_ids in stop_token_ids:
        for stop_ids in input_ids:
            #print("input_ids.shape :",input_ids.shape)
            print("*"*150)
            #print("input_ids.shape[0] :",input_ids[0].shape)
            #generated_tokens.append(input_ids)
            #print(tokenizer.decode(input_ids[0]))
            input_size = len(input_ids[0])
            print(tokenizer.decode(input_ids[0][input_size:]))



            #print("input_ids :",input_ids)
            #print("input_ids[0] :",input_ids[0])

            if torch.eq(input_ids[0][-len(stop_ids):], stop_ids).all():
                return True
        return False

pipeline = transformers.pipeline(
    "text-generation", #task
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.bfloat16,
    #trust_remote_code=True,
    device_map="auto",
    max_length=1500,
    do_sample=True,
    top_k=10,
    num_return_sequences=1,
    eos_token_id=tokenizer.eos_token_id,
    repetition_penalty=1.1,
    #stopping_criteria=stopping_criteria
)

prompt = "Explain Random Forest in 5 key points and maximum 100 words."
sequences = pipeline(
    prompt,
    do_sample=True,
    top_k=10,
    num_return_sequences=1,
    eos_token_id=tokenizer.eos_token_id,
    max_length=200,
)

'''
