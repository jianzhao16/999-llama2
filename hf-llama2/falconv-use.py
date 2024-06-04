'''
!pip install transformers==4.31.0
!pip install accelerate==0.21.0
!pip install bitsandbytes==0.40.2
!pip install peft==0.4.0
!pip install datasets
!pip install einops
!pip install torch==2.0.1
'''

import torch
from transformers import AutoTokenizer, FalconForCausalLM, BitsAndBytesConfig, TrainingArguments, Trainer, DataCollatorForLanguageModeling
import transformers
from peft import LoraConfig, get_peft_model, prepare_model_for_int8_training
from datasets import load_dataset, Dataset

model = "ybelkada/falcon-7b-sharded-bf16"

tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

bb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

falcon_model = FalconForCausalLM.from_pretrained(
    model,
    quantization_config=bb_config,
    use_cache=False
)


text = "Question: What is the national bird of the United States? \n Answer: "

inputs = tokenizer(text, return_tensors="pt").to("cuda:0")
outputs = falcon_model.generate(input_ids=inputs.input_ids, max_new_tokens=10)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))



text2 = "How do I make a HTML hyperlink?"

inputs = tokenizer(text2, return_tensors="pt").to("cuda:0")
outputs = falcon_model.generate(input_ids=inputs.input_ids, max_new_tokens=35)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))


text3 = "A 25-year-old female presents with swelling, pain, and inability to bear weight on her left ankle following a fall during a basketball game where she landed awkwardly on her foot. The pain is on the outer side of her ankle. What is the likely diagnosis and next steps? "

inputs = tokenizer(text3, return_tensors="pt").to("cuda:0")
outputs = falcon_model.generate(input_ids=inputs.input_ids, max_new_tokens=100)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))


