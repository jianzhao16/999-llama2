import os
import torch
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from transformers import LlamaForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset
from huggingface_hub import login

# Initialize Hugging Face authentication
hf_auth = "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"
#login(token=hf_auth)
login(token=hf_auth,add_to_git_credential=True)


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def train(rank, world_size):
    setup(rank, world_size)
    
    # Setup the device
    device = torch.device(f'cuda:{rank}')

    # Load the model and tokenizer
    model_name = 'meta-llama/llama-2-7b-chat-hf'
    model = LlamaForCausalLM.from_pretrained(model_name).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # Prepare the dataset
    dataset = load_dataset('wikitext', 'wikitext-2-raw-v1')

    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512)

    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    train_dataset = tokenized_datasets["train"]
    eval_dataset = tokenized_datasets["validation"]

    # Create a DistributedSampler and DataLoader for distributed training
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    train_loader = DataLoader(train_dataset, batch_size=2, sampler=train_sampler)

    eval_sampler = DistributedSampler(eval_dataset, num_replicas=world_size, rank=rank)
    eval_loader = DataLoader(eval_dataset, batch_size=2, sampler=eval_sampler)

    # Wrap the model with DistributedDataParallel
    model = DDP(model, device_ids=[rank])

    # Training arguments
    training_args = TrainingArguments(
        output_dir='./results',
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        num_train_epochs=1,
        logging_dir='./logs',
        logging_steps=10,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        fp16=True  # Enable mixed precision training
    )

    # Trainer initialization
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=lambda data: {'input_ids': torch.stack([f['input_ids'] for f in data]), 
                                    'attention_mask': torch.stack([f['attention_mask'] for f in data]), 
                                    'labels': torch.stack([f['input_ids'] for f in data])}
    )

    # Train the model
    trainer.train()

    # Save the fine-tuned model and tokenizer only from the main process
    if rank == 0:
        model.module.save_pretrained('./ddp_finetuned_model')
        tokenizer.save_pretrained('./ddp_finetuned_model')

    cleanup()

def main():
    world_size = torch.cuda.device_count()
    mp.spawn(train, args=(world_size,), nprocs=world_size, join=True)

if __name__ == "__main__":
    main()
