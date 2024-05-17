import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# model_path = 'ahxt/llama2_xs_460M_experimental'
model_path = 'ahxt/llama1_s_1.8B_experimental'

model = AutoModelForCausalLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)
model.eval()

prompt = 'Q: What is the largest bird?\nA:'
input_ids = tokenizer(prompt, return_tensors="pt").input_ids
tokens = model.generate(input_ids, max_length=20)
print( tokenizer.decode(tokens[0].tolist(), skip_special_tokens=True) )
# Q: What is the largest bird?\nA: The largest bird is the bald eagle.
