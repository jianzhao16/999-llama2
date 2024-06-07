from fastapi import FastAPI, Request
from transformers import AutoTokenizer, LlamaForCausalLM
import torch
from pydantic import BaseModel

app = FastAPI()

# Load the fine-tuned model and tokenizer
model_name_or_path = "./myllama2model-ourdata-v2"  # Update to your model path
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
model = LlamaForCausalLM.from_pretrained(model_name_or_path)
model.to("cuda" if torch.cuda.is_available() else "cpu")

class TextRequest(BaseModel):
    text: str
    max_new_tokens: int = 128

@app.post("/generate-text")
async def generate_text(request: TextRequest):
    inputs = tokenizer(request.text, return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")
    outputs = model.generate(input_ids=inputs.input_ids, max_new_tokens=request.max_new_tokens)
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return {"generated_text": generated_text}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
