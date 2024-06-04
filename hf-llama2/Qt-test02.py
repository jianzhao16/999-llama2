#from transformers import AutoModelForCausalLM, AutoTokenizer
from torch import cuda, bfloat16
from transformers import BitsAndBytesConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
#from bitsandbytes import Config  # or from transformers import BitsAndBytesConfig
import torch

#model = AutoModelForCausalLM.from_pretrained("model_name")
#tokenizer = AutoTokenizer.from_pretrained("model_name")

import torch
#torch.cuda.is_available()
# Output should be True


from transformers import AutoTokenizer
import transformers


access_token = "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"
model_name = "meta-llama/Llama-2-7b-hf"

bnb_config = transformers.BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=bfloat16
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map={"": 0},
    quantization_config=bnb_config,
)

model.config.use_cache = False
model.config.pretraining_tp = 1

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Ensure that the tokenizer is correctly initialized
if tokenizer is None:
    raise ValueError("Tokenizer not initialized correctly")

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Move the model to the GPU
#model = model.to(device)


pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
)

sequences = pipeline(
    'Hi! Tell me about yourself!',
    do_sample=True
)


'''
# Check if CUDA is available and set the device accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(model_name, token=access_token)

model = AutoModelForCausalLM.from_pretrained(
    model_name, 
    token=access_token
)


# Move the model to the GPU
model = model.to(device)

pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    torch_dtype=torch.float16,
    device_map="auto",
)

sequences = pipeline(
    'Hi! Tell me about yourself!',
    do_sample=True,
)
print(sequences[0].get("generated_text"))
'''
