import torch
from transformers import AutoTokenizer, FalconForCausalLM, BitsAndBytesConfig, TrainingArguments, Trainer, DataCollatorForLanguageModeling
import transformers
from peft import LoraConfig, get_peft_model, prepare_model_for_int8_training
from datasets import load_dataset, Dataset
import pandas as pd

# Initialize model and tokenizer
model_name = "ybelkada/falcon-7b-sharded-bf16"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

bb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

falcon_model = FalconForCausalLM.from_pretrained(
    model_name,
    quantization_config=bb_config
)

# Disable use_cache to be compatible with gradient checkpointing
falcon_model.config.use_cache = False

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

trainer = Trainer(
    model=lora_model,
    args=training_args,
    train_dataset=split_dataset["train"],
    eval_dataset=split_dataset["test"],
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
)

trainer.train()
trainer.model.save_pretrained("./finetuned_falcon")
