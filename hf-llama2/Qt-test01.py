from transformers import AutoModelForCausalLM, AutoTokenizer

#model = AutoModelForCausalLM.from_pretrained("model_name")
#tokenizer = AutoTokenizer.from_pretrained("model_name")

import torch
torch.cuda.is_available()
# Output should be True


from transformers import AutoTokenizer
import transformers


access_token = "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"
model_name = "meta-llama/Llama-2-7b-chat-hf"



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
