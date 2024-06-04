import torch
import transformers

from langchain.llms import HuggingFacePipeline

import time
#import torch

torch.cuda.empty_cache()  # Free up unused memory

start_time = time.time()  # Capture start time

# Set quantization configuration to load large model with less GPU memory
# This requires the `bitsandbytes` library
bnb_config = transformers.BitsAndBytesConfig(
    #load_in_4bit=False
    load_in_4bit=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16
)

#model_id = 'meta-llama/Llama-2-7b-hf'
hf_auth = "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"

model_id = "/home/tus35240/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3-8B/snapshots/62bd457b6fe961a42a631306577e622c83876cb6"

# Initialize the model and tokenizer
model_config = transformers.AutoConfig.from_pretrained(
    model_id,
    use_auth_token=hf_auth
)

model = transformers.AutoModelForCausalLM.from_pretrained(
    model_id,
    trust_remote_code=True,
    config=model_config,
    quantization_config=bnb_config,
    device_map='auto',
    use_auth_token=hf_auth
)
model.eval()

tokenizer = transformers.AutoTokenizer.from_pretrained(
    model_id,
    use_auth_token=hf_auth
)

# Set up the generation pipeline
generate_text = transformers.pipeline(
    model=model, tokenizer=tokenizer,
    return_full_text=True,
    task='text-generation',
    temperature=0.6,
    #max_new_tokens=512,
    repetition_penalty=1.1
)

end_time = time.time()  # Capture end time
elapsed_time = end_time - start_time  # Calculate elapsed time
print(f"Load Model Elapsed time: {elapsed_time} seconds")


start_time = time.time()  # Capture start time

# Example user interactions
examples = [
    "What is the square root of 16?",
    "Can you help me calculate the area of a circle with a radius of 5?",
    "What's 15% of 200?"
]

# Process each example
for example in examples:
    response = generate_text(example, max_length=50)
    print(f"User: {example}")
    print(f"Assistant: {response[0]['generated_text']}\n")


end_time = time.time()  # Capture end time
elapsed_time = end_time - start_time  # Calculate elapsed time
print(f"Inference Elapsed time: {elapsed_time} seconds")
