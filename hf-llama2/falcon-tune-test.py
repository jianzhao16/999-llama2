from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftConfig, PeftModel
import torch

# Step 1: Load the base model and tokenizer
falcon_model_name = "ybelkada/falcon-7b-sharded-bf16"  # Replace with your model name
print('load model begin')
falcon_model = AutoModelForCausalLM.from_pretrained(falcon_model_name).to("cuda:0")
tokenizer = AutoTokenizer.from_pretrained(falcon_model_name)

# Step 2: Load the fine-tuned configuration and model
config = PeftConfig.from_pretrained('./finetuned_falcon')
print('config done')

print('load fine tuned model begin')
finetuned_model = PeftModel.from_pretrained(falcon_model, './finetuned_falcon').to("cuda:0")
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
