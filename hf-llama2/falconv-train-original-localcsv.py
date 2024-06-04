# -*- coding: utf-8 -*-

# export PYTHONIOENCODING=utf-8
# python your_script.py
import torch
from transformers import AutoTokenizer, FalconForCausalLM, BitsAndBytesConfig, TrainingArguments, Trainer, DataCollatorForLanguageModeling
import transformers
from peft import LoraConfig, get_peft_model, prepare_model_for_int8_training
from datasets import load_dataset, Dataset
import pandas as pd

# Initialize model and tokenizer
#model_name = "ybelkada/falcon-7b-sharded-bf16"


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


text3 = "A 25-year-old female presents with swelling, pain, and inability to bear weight on her left ankle following a fall during a basketball game where she landed awkwardly on her foot. The pain is on the outer side of her ankle. What is the likely diagnosis and next steps? "

inputs = tokenizer(text3, return_tensors="pt").to("cuda:0")
outputs = falcon_model.generate(input_ids=inputs.input_ids, max_new_tokens=100)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))


# Your input text
text_pad1 = "Question: What is the national bird of the United States? \n Answer: "

# Tokenize the input
inputs = tokenizer(text_pad1, return_tensors="pt", padding=True).to("cuda:0")

# Set the attention mask
attention_mask = inputs["attention_mask"]

# Generate output with the specified attention mask and pad token id
outputs = falcon_model.generate(
    inputs["input_ids"],
    attention_mask=attention_mask,
    pad_token_id=tokenizer.eos_token_id
)

# Decode the generated output
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(generated_text)

'''


text = "Question: What is the national bird of the United States? \n Answer: "

inputs = tokenizer(text, return_tensors="pt").to("cuda:0")
outputs = falcon_model.generate(input_ids=inputs.input_ids, max_new_tokens=10)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))

text2 = "How do I make a HTML hyperlink?"

inputs = tokenizer(text2, return_tensors="pt").to("cuda:0")
outputs = falcon_model.generate(input_ids=inputs.input_ids, max_new_tokens=35)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
'''






training_args = TrainingArguments(
    output_dir="./finetuned_falcon",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    weight_decay=0.01,
    fp16=True,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=1,
    logging_steps=1,
    num_train_epochs=1,
    optim="paged_adamw_8bit"
)


print("Preparing Falcon model")
falcon_model.gradient_checkpointing_enable()
falcon_model = prepare_model_for_int8_training(falcon_model)
print("Preparing Lora model parameters")
lora_config = LoraConfig(
    r=4,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[
        "query_key_value",
        "dense",
        "dense_h_to_4h",
        "dense_4h_to_h",
    ]
)


lora_model = get_peft_model(falcon_model, lora_config)

print("Getting PEFT Lora model")
lora_model = get_peft_model(falcon_model, lora_config)
print("PEFT model Lora done")

def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )

print_trainable_parameters(lora_model)


import pandas as pd
#dataset = load_dataset("BI55/MedText", split="train")
#df = pd.DataFrame(dataset)
file_path = './ourdata-ppd.csv'
df = pd.read_csv(file_path)

prompt = df.pop("Prompt")
comp = df.pop("Completion")
df["Info"] = prompt + "\n" + comp

max_length_set = 158
# https://www.kaggle.com/code/harveenchadha/tokenize-train-data-using-bert-tokenizer
def tokenizing(text, tokenizer, chunk_size, maxlen):
    input_ids = []
    tt_ids = []
    at_ids = []

    for i in range(0, len(text), chunk_size):
        text_chunk = text[i:i+chunk_size]
        encs = tokenizer(
                    text_chunk,
                    max_length = max_length_set,
                    padding='max_length',
                    truncation=True
                    )

        input_ids.extend(encs['input_ids'])
        tt_ids.extend(encs['token_type_ids'])
        at_ids.extend(encs['attention_mask'])

    return {'input_ids': input_ids, 'token_type_ids': tt_ids, 'attention_mask':at_ids}

# Tokenize the dataset
tokens = tokenizing(list(df["Info"]), tokenizer, 256, max_length_set)
tokens_dataset = Dataset.from_dict(tokens)
split_dataset = tokens_dataset.train_test_split(test_size=0.2)
split_dataset

print(split_dataset)

trainer = Trainer(
    model=lora_model,
    args=training_args,
    train_dataset=split_dataset["train"],
    eval_dataset=split_dataset["test"],
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
)

trainer.train()
trainer.model.save_pretrained("./finetuned_falcon")


print('Test Begin')
from peft import PeftConfig, PeftModel

config = PeftConfig.from_pretrained('./finetuned_falcon')
finetuned_model = PeftModel.from_pretrained(falcon_model, './finetuned_falcon')

text4 = "A 25-year-old female presents with swelling, pain, and inability to bear weight on her left ankle following a fall during a basketball game where she landed awkwardly on her foot. The pain is on the outer side of her ankle. What is the likely diagnosis and next steps?"

inputs = tokenizer(text4, return_tensors="pt").to("cuda:0")
outputs = finetuned_model.generate(input_ids=inputs.input_ids, max_new_tokens=75)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
