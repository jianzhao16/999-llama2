import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def load_model(model_id, device='cuda'):
    # Ensure that the model and tokenizer are loaded
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    print('load model begin')
    model = AutoModelForCausalLM.from_pretrained(model_id)

    # Move the model to the specified device
    model.to(device)
    
    # Set the model to evaluation mode
    model.eval()

    print('load model done')

    return model, tokenizer

# Example usage
#model_id = 'metal-llama/Meta-Llama-3-8B'  # Hypothetical model ID; adjust as needed
model_id = 'metal-llama/Llama-2-7b-hf'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model, tokenizer = load_model(model_id, device)
print('load all done')
