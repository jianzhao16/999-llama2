from peft import PeftConfig, PeftModel
from transformers import LlamaForCausalLM, AutoTokenizer, BitsAndBytesConfig
from huggingface_hub import login
import torch
import os

# Initialize Hugging Face authentication
hf_auth = "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"
login(token=hf_auth, add_to_git_credential=True)

model_tuned_path = './myllama2model'  # Ensure this path is correct and accessible

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

tokenizer = AutoTokenizer.from_pretrained(llama2orginal)

# Step 2: Load the fine-tuned configuration and model
print('Loading fine-tuned model configuration...')
config_path = os.path.join(model_tuned_path, 'adapter_config.json')
if not os.path.isfile(config_path):
    raise FileNotFoundError(f"Configuration file not found at {config_path}")

with open(config_path, 'r') as f:
    config_data = f.read()
    print(f"Config Data: {config_data}")

config = PeftConfig.from_pretrained(model_tuned_path)
print('Configuration loaded.')

print('Loading fine-tuned model...')
try:
    finetuned_model = PeftModel.from_pretrained(model_orginal, model_tuned_path).to("cuda:0")
    print('Fine-tuned model loaded successfully.')
except KeyError as e:
    print(f"Error loading fine-tuned model: {e}")
    exit()

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
