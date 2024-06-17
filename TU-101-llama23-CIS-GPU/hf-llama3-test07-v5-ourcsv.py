# -*- coding: utf-8 -*-
from datetime import time

# export PYTHONIOENCODING=utf-8
# python your_script.py
from transformers import LlamaForCausalLM, AutoTokenizer, FalconForCausalLM, BitsAndBytesConfig, TrainingArguments, \
    Trainer, DataCollatorForLanguageModeling
import transformers
from peft import LoraConfig, get_peft_model, prepare_model_for_int8_training
from datasets import load_dataset, Dataset
import pandas as pd
import torch

from huggingface_hub import login

import os
from collections import deque
import time
from dotenv import load_dotenv


print('init')
torch.cuda.empty_cache()

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
max_length_set = 256

print('load env')
# Initialize Hugging Face authentication
# Load .env file
load_dotenv()
# Retrieve API key
hf_auth = os.getenv("HF_wr")

print('login ....')
login(token=hf_auth,add_to_git_credential=True)

# Path to the model
#model = "meta-llama/Meta-Llama-3-8B-Instruct"
model  = 'Base-Llama-3-8B-Instruct'
max_new_tokens=128
#model = 'Base-Llama-2-7b-chat-hf'
print(model)
# model_name = "/home/tus35240/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3-8B/snapshots/62bd457b6fe961a42a631306577e622c83876cb6"

# model_name = "meta-llama/Llama-2-7B-chat-hf"


# model = "/home/tus35240/.cache/huggingface/hub/models--meta-llama--Llama-2-7b-chat-hf/snapshots/f5db02db724555f92da89c216ac04704f23d4590"
# model = "ybelkada/falcon-7b-sharded-bf16"

tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

print('tokenizer load')

bb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

print("bb_config load is done")

falcon_model = LlamaForCausalLM.from_pretrained(
    model,
    quantization_config=bb_config,
    low_cpu_mem_usage=True,
    use_cache=False
)
print("model load is done")

# text3 = "A 25-year-old female presents with swelling, pain, and inability to bear weight on her left ankle following a fall during a basketball game where she landed awkwardly on her foot. The pain is on the outer side of her ankle. What is the likely diagnosis and next steps? "

# inputs = tokenizer(text3, return_tensors="pt").to("cuda:0")
# outputs = falcon_model.generate(input_ids=inputs.input_ids, max_new_tokens=100)
# print(tokenizer.decode(outputs[0], skip_special_tokens=True))


print('begin test some inputs in base model')
print('**************************************')
# Your input text
#text_pad1 = "Question: What is the national bird of the United States? \n Answer: "
text_pad1 = "A 25-year-old female presents with swelling, pain, and inability to bear weight on her left ankle following a fall during a basketball game where she landed awkwardly on her foot. The pain is on the outer side of her ankle. What is the likely diagnosis and next steps? "
# Tokenize the input
inputs = tokenizer(text_pad1, return_tensors="pt", padding=True).to("cuda:0")

# Set the attention mask
attention_mask = inputs["attention_mask"]

# Generate output with the specified attention mask and pad token id
outputs = falcon_model.generate(
    inputs["input_ids"],
    attention_mask=attention_mask,
    max_new_tokens=100,
    pad_token_id=tokenizer.eos_token_id
)

# Decode the generated output
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(generated_text)

print('test is done')
print('*****************************************')

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
'''


# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=1,  # Reduce batch size
    per_device_eval_batch_size=1,  # Reduce evaluation batch size
    gradient_accumulation_steps=4,  # Accumulate gradients to simulate larger batch size
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    evaluation_strategy="steps",
    save_steps=10_000,
    eval_steps=500,
    save_total_limit=2,
    fp16=True,  # Enable mixed precision training
    dataloader_pin_memory=False,  # Disable pin memory if using torch.bfloat16
)

print("Preparing gradient model")
falcon_model.gradient_checkpointing_enable()
falcon_model = prepare_model_for_int8_training(falcon_model)
print("Preparing Lora model parameters")

'''
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

lora_model = falcon_model

from peft import LoraConfig

peft_config = LoraConfig(
    lora_alpha=32,
    lora_dropout=0.05,
    r=4,
    bias="none",
    task_type="CAUSAL_LM",
)

'''

lora_config = LoraConfig(
    r=4,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)


lora_model = get_peft_model(falcon_model, lora_config)
print("PEFT model Lora done")

#lora_model.add_adapter(peft_config)


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

# https://www.kaggle.com/code/harveenchadha/tokenize-train-data-using-bert-tokenizer
'''
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
'''


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
        #tt_ids.extend(encs['token_type_ids'])
        at_ids.extend(encs['attention_mask'])

    #return {'input_ids': input_ids, 'token_type_ids': tt_ids, 'attention_mask':at_ids}
    return {'input_ids': input_ids, 'attention_mask': at_ids}

# Tokenize the dataset
tokens = tokenizing(list(df["Info"]), tokenizer, 256, max_length_set)

# def tokenizingv2(texts, tokenizer, pad_to_max_length, max_length_set):
#     encs = tokenizer.batch_encode_plus(
#         texts,
#         max_length=max_length_set,
#         padding='max_length' if pad_to_max_length else False,
#         truncation=True,
#         return_tensors="pt",
#         return_token_type_ids=True  # Ensure this is set to True if needed
#     )
#
#     input_ids = encs['input_ids']
#     attention_mask = encs['attention_mask']
#     token_type_ids = encs.get('token_type_ids', None)  # Use get method to avoid KeyError
#
#     return input_ids, attention_mask, token_type_ids

# Example usage
#tokens, masks, token_types = tokenizing(list(df["Info"]), tokenizer, pad_to_max_length=True, max_length_set=256)

'''
# Handle cases where token_type_ids might be None
if token_types is not None:
    tt_ids.extend(token_types)
else:
    tt_ids = None  # Or handle the absence of token_type_ids as needed
'''


# Convert the tensor to a dictionary
#tokens_dict = {'input_ids': tokens.tolist()}

# Create the dataset
#tokens_dataset = Dataset.from_dict(tokens_dict)


print('tokens is done')
#print(tokens)
print('*****************************************')

tokens_dataset = Dataset.from_dict(tokens)

print(tokens_dataset)
print('*****************************************')

split_dataset = tokens_dataset.train_test_split(test_size=0.2)
split_dataset

print(split_dataset)


print('*****************************************')
trainer = Trainer(
    model=lora_model,
    args=training_args,
    train_dataset=split_dataset["train"],
    eval_dataset=split_dataset["test"],
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
)

print('*****************************************')
print('trainer is begin')
trainer.train()
print('trainer is done')

#trainer.model.save_pretrained("./finetuned_falcon")
print('save begin')
# Define the path where you want to save the model
output_dir = "./myllama3model-ourdata"

# Save the model weights using torch.save
model_to_save = trainer.model.module if hasattr(trainer.model, 'module') else trainer.model
torch.save(model_to_save.state_dict(), f"{output_dir}/pytorch_model.bin")

# Save the tokenizer and configuration separately
#tokenizer = AutoTokenizer.from_pretrained("your-model-name")
tokenizer.save_pretrained(output_dir)

# Save the model configuration
model_to_save.config.save_pretrained(output_dir)

print(f"Model saved to {output_dir} done")



output_dir_v2 = "./myllama3model-ourdata-v2"
print('Saving the model and tokenizer...')
model_to_save.save_pretrained(output_dir_v2)
tokenizer.save_pretrained(output_dir_v2)
print(f"Model saved to {output_dir_v2} done")

#trainer.model.save_pretrained("./finetuned_falcon")
#trainer.save_model("./myllama2model")
#print('Model is saved')


from peft import PeftConfig, PeftModel

#config = PeftConfig.from_pretrained('./finetuned_falcon')
#finetuned_model = PeftModel.from_pretrained(falcon_model, './finetuned_falcon')

finetuned_model = trainer.model

text4 = "A 25-year-old female presents with swelling, pain, and inability to bear weight on her left ankle following a fall during a basketball game where she landed awkwardly on her foot. The pain is on the outer side of her ankle. What is the likely diagnosis and next steps?"
inputs = tokenizer(text4, return_tensors="pt").to("cuda:0")
outputs = finetuned_model.generate(input_ids=inputs.input_ids, max_new_tokens=75)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))

text5 = "where is the food service nearby?"
inputs = tokenizer(text5, return_tensors="pt").to("cuda:0")
outputs = finetuned_model.generate(input_ids=inputs.input_ids, max_new_tokens=75)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))


class ConversationMemory:
    def __init__(self):
        # Set maxlen to 2 to keep only the most recent two entries
        self.history = deque([], maxlen=2)

    def update_user_input(self, input_text):
        # Truncate the text to 20% if it is too long
        if len(input_text) > 100:  # Assuming 'too long' means more than 100 characters
            input_text = input_text[:int(len(input_text) * 0.4)]
        self.history.append({"role": "user", "content": input_text})

    def update_assistant_response(self, response_text):
        # Truncate the text to 20% if it is too long
        if len(response_text) > 100:  # Assuming 'too long' means more than 100 characters
            response_text = response_text[:int(len(response_text) * 0.3)]
        self.history.append({"role": "assistant", "content": response_text})

    def get_history(self):
        return list(self.history)

    def clear_history(self):
        self.history.clear()

    def display_history(self):
        print("Conversation History:")
        for item in self.history:
            time = item['timestamp']
            role = item['role'].capitalize()
            content = item['content']
            print(f"{time} - {role}: {content}")

    def get_total_content_length(self):
        # Calculate the total length of all content in the history
        total_length = sum(len(item['content']) for item in self.history)
        return total_length

    def to_json(self):
        import json
        return json.dumps(list(self.history), indent=4)

# Step 3: Use the fine-tuned model
def generate_text(model, tokenizer, text, max_new_tokens=128):
    inputs = tokenizer(text, return_tensors="pt").to("cuda:0")
    outputs = model.generate(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask, max_new_tokens=max_new_tokens)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

print('Begin testing fine-tuned model')


# while True:
#     text = input("Enter text (or type 'exit' to quit): ")
#     if text.lower() == 'exit':
#         break
#     output_text = generate_text(finetuned_model, tokenizer, text)
#     print(f"Output: {output_text}")
#generator = Llama.build(ckpt_dir=ckpt_dir, tokenizer_path=tokenizer_path, max_seq_len=max_seq_len,
#                        max_batch_size=max_batch_size)

# last_input = ""

#ckpt_dir: str,
#tokenizer_path: str,
temperature = 0.6
top_p = 0.9,
max_seq_len = 256
max_batch_size = 8
max_gen_len  = None

# Initialize the conversation memory
conversation = ConversationMemory()

try:
    while True:
        # Initialize an empty list to store the conversation history
        # dialogs is  [ [] ] 2-dimensional list
        dialogs = []
        print("\n**************************************\n")
        current_input = input("Enter your instruction (type 'exit' to finish): ")

        if current_input.lower() == "exit":
            break
        if current_input == "":
            print('Empty instruction, please try again.')
            continue

        # dialogs.append([{"role": "user", "content": current_input}])

        # last_input = current_input

        # Update the conversation with the user's latest input
        conversation.update_user_input(current_input)

        # Generate a prompt using the conversation history and memory
        conversation_input = conversation.get_history()

        #print(f" Conversation Prompt: {conversation_input}")
        print(f" Prompt: {conversation.get_total_content_length()}")

        # dialogs.append([{"role": "user", "content": prompt_input}])
        dialogs.append(conversation_input)

        start_time = time.time()

        print('Begin Inference...')
        #results = finetuned_model.chat_completion([dialogs[-1]], max_gen_len=max_gen_len, temperature=temperature, top_p=top_p)

        # print('dialogs 2 dimensional list:')
        # print([dialogs[-1]])
        #
        # print('dialogs last element:')
        # print(dialogs[-1])

        last_dialog = dialogs[-1]  # This gives you the last element in the outer list, which is another list
        last_message = last_dialog[-1]  # This gives you the last element in the inner list, which is a dictionary
        last_message_content = last_message['content']  # This accesses the 'content' key in the dictionary


        #output_text = generate_text(finetuned_model, tokenizer, last_message_content, max_new_tokens=128)
        inputs = tokenizer(last_message_content, return_tensors="pt").to("cuda:0")
        outputs = finetuned_model.generate(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask,
                                 max_new_tokens=max_new_tokens)
        output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        #print('ourput_text:', output_text)

        results = [{'generation': {'role': 'assistant', 'content': output_text}}]

        # results = generator.chat_completion(conversation_input[-1], max_gen_len=max_gen_len, temperature=temperature, top_p=top_p)
        print('Inference Results:')
        #print(results)
        print(" results length:")
        print(len(results))

        # results = [{'generation': {'role': 'assistant', 'content': "I'm happy."}}]

        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Inference Elapsed time: {elapsed_time:.2f} seconds")

        # Update the conversation with the generated response
        # print('Results:',results)
        # print('Results[0]:',results[-1]['generation']['content'])

        # Update the conversation with the generated response
        conversation.update_assistant_response(results[0]['generation']['content'])
        # dialogs.append([{"role": "assistant", "content": results[0]['generation']['content']}])

        # print(dialogs[-1])
        print("\n==================================\n")
        print("Result Display: ")
        # Display the conversation history
        print("\n============== All History Display====================\n")
        print(f"Conversation Memory: {conversation.display_history}")
        # print(f"Conversation Memory: {conversation.get_conversation_prompt()}")

        print("\n==============Cureent Answer====================\n")
        print("Results")
        # print(results)
        print(f"> {results[0]['generation']['role'].capitalize()}: {results[0]['generation']['content']}")
        '''
        for dialog, result in zip([dialogs[-1]], results):
            for msg in dialog:
                print(f"{msg['role'].capitalize()}: {msg['content']}\n")
            print(f"> {result['generation']['role'].capitalize()}: {result['generation']['content']}")
           print("\n==================================\n")
        '''

except KeyboardInterrupt:
    print("Interrupted by user.")

print("Session Ended.")

