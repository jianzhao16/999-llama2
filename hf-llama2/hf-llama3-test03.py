import transformers

# Ensure you have the model ID correctly specified
#model_id = "meta-llama/Llama-2-7b-hf"

## Download 8B model
## huggingface-cli login
## input token
## huggingface-cli download meta-llama/Meta-Llama-3-8B --include "original/*" --local-dir Meta-Llama-3-8B
## huggingface-cli repo create meta-llama/Meta-Llama-3-8B
## model_id = 'meta-llama/Llama-3-8b-hf'

#model_id = "meta-llama/Llama-2-7b-hf"
model_id = "/home/tus35240/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3-8B/snapshots/62bd457b6fe961a42a631306577e622c83876cb6"
hf_auth = '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'


# Initialize the pipeline with the specified model for text generation
pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    tokenizer=model_id,  # Generally, the tokenizer ID is the same as the model ID
    device=0  # Specify the device; use -1 for CPU, or a GPU index like 0
)

# Generate text using the pipeline
response = pipeline("Hey how are you doing today?", max_length=50)  # You can adjust max_length as needed

# Print the generated text
print(response[0]['generated_text'])
