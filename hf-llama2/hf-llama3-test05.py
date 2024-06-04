import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
import torch
from transformers import AutoTokenizer, FalconForCausalLM, BitsAndBytesConfig, TrainingArguments, Trainer, \
    DataCollatorForLanguageModeling
import transformers
from peft import LoraConfig, get_peft_model
from datasets import load_dataset, Dataset
import pandas as pd

# Initialize model and tokenizer
# model_name = "ybelkada/falcon-7b-sharded-bf16"

hf_auth = "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"

model_name = "/home/tus35240/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3-8B/snapshots/62bd457b6fe961a42a631306577e622c83876cb6"

#model_name = "meta-llama/Meta-Llama-3-8B-Instruct"


from huggingface_hub import login

login(token=hf_auth)



import requests
import json
import pandas as pd
from transformers import PreTrainedTokenizerFast
from datasets import Dataset


import json
import pandas as pd
from transformers import PreTrainedTokenizerFast
from datasets import Dataset
import os

def process_dataset(file_path, config_path, chunk_size=256, max_length=512):
    # Load the tokenizer configuration JSON from the local file
    if os.path.isfile(config_path):
        with open(config_path, 'r') as f:
            tokenizer_config = json.load(f)
    else:
        raise FileNotFoundError(f"The file {config_path} does not exist.")

    # Initialize the tokenizer using the configuration
    tokenizer = PreTrainedTokenizerFast(**tokenizer_config)

    # Load custom dataset
    df = pd.read_csv(file_path)

    prompt = df.pop("Prompt")
    comp = df.pop("Completion")
    df["Info"] = prompt + "\n" + comp
    print(df)

    def tokenizing(text, tokenizer, chunk_size, max_length):
        input_ids = []
        attention_masks = []

        for i in range(0, len(text), chunk_size):
            text_chunk = text[i:i+chunk_size]
            encodings = tokenizer(
                text_chunk,
                max_length=max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )

            input_ids.extend(encodings['input_ids'].tolist())
            attention_masks.extend(encodings['attention_mask'].tolist())

        return {'input_ids': input_ids, 'attention_mask': attention_masks}

    print("Tokenizing dataset")
    tokens = tokenizing(list(df["Info"]), tokenizer, chunk_size, max_length)
    tokens_dataset = Dataset.from_dict(tokens)
    print(tokens_dataset)
    print('Tokenizing done')

    return tokens_dataset

# Usage example
file_path = "./medtext_2.csv"
config_path = "./llama3_tokenizer_config.json"
tokens_dataset = process_dataset(file_path, config_path)



'''
# Step 1: Load the Dataset
file_path = './medtext_2.csv'
data = pd.read_csv(file_path)

# Convert to Hugging Face dataset
hf_dataset = Dataset.from_pandas(data)

# Step 2: Tokenize the Dataset
# tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Assign a padding token if it doesn't exist
#if tokenizer.pad_token is None:
#    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

# Set a maximum length for truncation and padding
max_length = 512


def tokenize_function(examples):
    return tokenizer(examples['Prompt'], padding='max_length', truncation=True, max_length=max_length)


tokenized_dataset = hf_dataset.map(tokenize_function, batched=True)




tokenized_dataset = hf_dataset.map(tokenize_function, batched=True)
'''

#print("get tokenizer")
#tokenizer = AutoTokenizer.from_pretrained(model_name)

#tokenizer = AutoTokenizer.from_pretrained("Meta-Llama-3-8B/original/tokenizer.model")

# Step 2: Tokenize the Dataset
# tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
#tokenizer = AutoTokenizer.from_pretrained(model_name)

# Assign a padding token if it doesn't exist
# if tokenizer.pad_token is None:
#     print('PAD begin')
#     tokenizer.add_special_tokens({'pad_token': '[PAD]'})

# Set a maximum length for truncation and padding


#tokenizer.pad_token = tokenizer.eos_token
#max_length = 512


#from transformers import LlamaTokenizer

# Initialize the LLama 3 8B tokenizer
#tokenizer = LlamaTokenizer.from_pretrained('/home/tus35240/mydata/mygit/llama3/Meta-Llama-3-8B/tokenizer.model')
t#okenizer = AutoTokenizer.from_pretrained(model_name)

#print(tokenizer)
#
# def tokenizing(text, tokenizer, chunk_size, max_length):
#     input_ids = []
#     attention_masks = []
#
#     for i in range(0, len(text), chunk_size):
#         text_chunk = text[i:i+chunk_size]
#         encodings = tokenizer(
#             text_chunk,
#             max_length=max_length,
#             padding='max_length',
#             truncation=True,
#             return_tensors='pt'
#         )
#
#         input_ids.extend(encodings['input_ids'].tolist())
#         attention_masks.extend(encodings['attention_mask'].tolist())
#
#     return {'input_ids': input_ids, 'attention_mask': attention_masks}
#
# print("Tokenizing dataset")
# tokens = tokenizing(list(df["Info"]), tokenizer, chunk_size=256, max_length=512)
# tokens_dataset = Dataset.from_dict(tokens)
# print(tokens_dataset)
# print('Tokenizing done')

'''
print("load dataset")
dataset = load_dataset("BI55/MedText", split="train")
import pandas as pd

df = pd.DataFrame(dataset)
prompt = df.pop("Prompt")
comp = df.pop("Completion")
df["Info"] = prompt + "\n" + comp
print(df)

def tokenizing(text, tokenizer, chunk_size, maxlen):
    input_ids = []
    tt_ids = []
    at_ids = []

    for i in range(0, len(text), chunk_size):
        text_chunk = text[i:i+chunk_size]
        encs = tokenizer(
                    text_chunk,
                    max_length = max_length,
                    padding='max_length',
                    truncation=True
                    )

        input_ids.extend(encs['input_ids'])
        tt_ids.extend(encs['token_type_ids'])
        at_ids.extend(encs['attention_mask'])

    return {'input_ids': input_ids, 'token_type_ids': tt_ids, 'attention_mask':at_ids}

print("Tokenizing dataset")
tokens = tokenizing(list(df["Info"]), tokenizer, 256, max_length)
tokens_dataset = Dataset.from_dict(tokens)
print(tokens_dataset)
print('Tokenizing done')
'''

# Step 3: Split the Dataset
print("Splitting dataset")
split_dataset = tokens_dataset.train_test_split(test_size=0.2)

# Split the dataset into train and test sets
#split_dataset = tokenized_dataset.train_test_split(test_size=0.2)

bb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

falcon_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bb_config
)

# Disable use_cache to be compatible with gradient checkpointing
falcon_model.config.use_cache = False

# Step 4: Set Training Arguments
training_args = TrainingArguments(
    output_dir='./results',
    overwrite_output_dir=True,
    num_train_epochs=1,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
)

# Step 5: Initialize the Trainer
trainer = Trainer(
    model=falcon_model,
    args=training_args,
    train_dataset=split_dataset["train"],
    eval_dataset=split_dataset["test"],
    data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
)

# Step 6: Train the Model
trainer.train()

# Step 7: Evaluate the Model
results = trainer.evaluate()
print(results)

# Step 8: Save the Model
model.save_pretrained('savemodel')
tokenizer.save_pretrained('savetokenizer')

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
prepare_model_for_kbit_training(falcon_model, 8)

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

print("Loading dataset")
pathcsv = "./ourdata.csv"
df = pd.read_csv(pathcsv)

prompt = df.pop("Prompt")
comp = df.pop("Completion")
df["Info"] = prompt + "\n" + comp

def tokenizing(texts, tokenizer, maxlen):
    encodings = tokenizer(
        texts,
        max_length=maxlen,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    return encodings

# Tokenize the dataset
print("Tokenizing dataset")
tokens = tokenizing(list(df["Info"]), tokenizer, 2048)

# Convert tokens to a Hugging Face dataset
tokens_dataset = Dataset.from_dict({
    'input_ids': tokens['input_ids'],
    'attention_mask': tokens['attention_mask']
})

# Split the dataset into train and test sets
split_dataset = tokens_dataset.train_test_split(test_size=0.2)

print(split_dataset)
print('split datasets done')

print('Begin of Traing')
trainer = Trainer(
    model=lora_model,
    args=training_args,
    train_dataset=split_dataset["train"],
    eval_dataset=split_dataset["test"],
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
)

trainer.train()

print('Trainning is done')
trainer.model.save_pretrained("./finetuned_falcon")
'''
