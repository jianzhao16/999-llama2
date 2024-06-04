from peft import PeftConfig, PeftModel
from transformers import LlamaForCausalLM, AutoTokenizer, BitsAndBytesConfig
from huggingface_hub import login
import torch

# Initialize Hugging Face authentication
hf_auth = "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"
login(token=hf_auth, add_to_git_credential=True)

# Model name and paths
model_name = 'meta-llama/llama-2-7b-chat-hf'
local_model_path = './myllama2model'  # Ensure this path is correct and accessible

# Load the tokenizer
print('Loading tokenizer...')
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

# BitsAndBytes configuration for 4-bit quantization
bb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

print("Loading model...")
# Load the base model with quantization
model = LlamaForCausalLM.from_pretrained(
    model_name,
    quantization_config=bb_config,
    use_cache=False
)

print("Model loaded successfully.")

# Load the fine-tuned configuration and model
print('Loading fine-tuned model configuration...')
try:
    config = PeftConfig.from_pretrained(local_model_path)
    print('Configuration loaded.')

    print('Loading fine-tuned model...')
    finetuned_model = PeftModel.from_pretrained(model, local_model_path).to("cuda:0")
    print('Fine-tuned model loaded successfully.')
except Exception as e:
    print(f"Error loading configuration or model: {e}")
    exit()

# Generate text using the fine-tuned model
def generate_text(model, tokenizer, text, max_new_tokens=75):
    inputs = tokenizer(text, return_tensors="pt").to("cuda:0")
    outputs = model.generate(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask, max_new_tokens=max_new_tokens)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Example usage
input_text1 = "Hello, world!"
output_text1 = generate_text(finetuned_model, tokenizer, input_text1)
print(f"Output: {output_text1}")

input_text2 = "A 25-year-old female presents with swelling, pain, and inability to bear weight on her left ankle following a fall."
output_text2 = generate_text(finetuned_model, tokenizer, input_text2)
print(f"Output: {output_text2}")
