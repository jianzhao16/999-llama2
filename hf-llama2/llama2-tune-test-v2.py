# -*- coding: utf-8 -*-

# export PYTHONIOENCODING=utf-8


import transformers
from peft import LoraConfig, get_peft_model, prepare_model_for_int8_training
import os
from peft import PeftConfig, PeftModel
from transformers import LlamaForCausalLM, AutoTokenizer, BitsAndBytesConfig,AutoModelForCausalLM
from huggingface_hub import login
import torch


torch.cuda.empty_cache()

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
max_length_set = 256

# Initialize Hugging Face authentication
hf_auth = "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"
login(token=hf_auth, add_to_git_credential=True)

model_tuned_path = 'myllama2model'  # Ensure this path is correct and accessible



base_model_id = "meta-llama/Llama-2-7b-chat-hf"
#model = 'meta-llama/llama-2-7b-chat-hf'
# model_name = "/home/tus35240/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3-8B/snapshots/62bd457b6fe961a42a631306577e622c83876cb6"

# model_name = "meta-llama/Llama-2-7B-chat-hf"


# model = "/home/tus35240/.cache/huggingface/hub/models--meta-llama--Llama-2-7b-chat-hf/snapshots/f5db02db724555f92da89c216ac04704f23d4590"
# model = "ybelkada/falcon-7b-sharded-bf16"

tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

bb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

print("bb_config load is done")

llama2_4bit_model = LlamaForCausalLM.from_pretrained(
    base_model_id,
    quantization_config=bb_config,
    use_cache=False
)
print("model load is done")


print("Preparing llama2 model")
llama2_4bit_model.gradient_checkpointing_enable()
llama2_4bit_model = prepare_model_for_int8_training(llama2_4bit_model)
print('llama2 model done')


finetuned_model = AutoModelForCausalLM.from_pretrained(llama2_4bit_model, model_tuned_path).to("cuda:0")
#finetuned_model = LlamaForCausalLM.from_pretrained(llama2_4bit_model, model_tuned_path).to("cuda:0")

#finetuned_model = LlamaForCausalLM.from_pretrained(model_tuned_path).to("cuda:0")
print('load fine tuned model done')


print('Begin test')
text4 = "A 25-year-old female presents with swelling, pain, and inability to bear weight on her left ankle following a fall."
inputs = tokenizer(text4, return_tensors="pt").to("cuda:0")
attention_mask = inputs.attention_mask.to("cuda:0")
outputs = finetuned_model.generate(input_ids=inputs.input_ids, attention_mask=attention_mask, max_new_tokens=75)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))

print('End test')
'''
# Step 1: Load the base model and tokenizer
llama2orginal = 'meta-llama/llama-2-7b-chat-hf'  # Replace with your model name
print('load model begin')

# BitsAndBytes configuration for 4-bit quantization
bb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

print("Loading model...")
# Load the base model with quantization
model_orginal = LlamaForCausalLM.from_pretrained(
    llama2orginal,
    quantization_config=bb_config,
    use_cache=False
)

#falcon_model = LlamaForCausalLM.from_pretrained(llama2orginal).to("cuda:0")
tokenizer = AutoTokenizer.from_pretrained(llama2orginal)

# Step 2: Load the fine-tuned configuration and model
config = PeftConfig.from_pretrained(model_tuned_path)
print('config done')

print('load fine tuned model begin')
finetuned_model = PeftModel.from_pretrained(model_orginal, model_tuned_path).to("cuda:0")
print('load fine tuned model done')

# Step 3: Use the finetuned model
# Example: Tokenize some input text
input_text = "Hello, world!"
inputs = tokenizer(input_text, return_tensors="pt").to("cuda:0")
attention_mask = inputs.attention_mask.to("cuda:0")
outputs = finetuned_model.generate(input_ids=inputs.input_ids, attention_mask=attention_mask, max_new_tokens=75)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))

text4 = "A 25-year-old female presents with swelling, pain, and inability to bear weight on her left ankle following a fall."
inputs = tokenizer(text4, return_tensors="pt").to("cuda:0")
attention_mask = inputs.attention_mask.to("cuda:0")
outputs = finetuned_model.generate(input_ids=inputs.input_ids, attention_mask=attention_mask, max_new_tokens=75)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
'''