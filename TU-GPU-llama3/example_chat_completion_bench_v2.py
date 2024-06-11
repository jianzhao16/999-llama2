# -*- coding: utf-8 -*-
from collections import deque
import fire
import time
import torch
from llama import Llama, Dialog

def check_gpu_memory():
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        print(f"Number of GPUs available: {num_gpus}")
        for gpu_id in range(num_gpus):
            torch.cuda.set_device(gpu_id)
            total_memory = torch.cuda.get_device_properties(gpu_id).total_memory
            free_memory = total_memory - torch.cuda.memory_allocated(gpu_id)
            total_memory_gb = total_memory / (1024 ** 3)
            free_memory_gb = free_memory / (1024 ** 3)
            print(f"GPU {gpu_id}: Total Memory: {total_memory_gb:.2f} GB, Free Memory: {free_memory_gb:.2f} GB")
    else:
        print("CUDA is not available. Check your installation or GPU availability.")

def similarity(input1, input2):
    set1 = set(input1.split())
    set2 = set(input2.split())
    return len(set1 & set2) > len(set1) / 2

def main(ckpt_dir, tokenizer_path, temperature=0.6, top_p=0.9, max_seq_len=512, max_batch_size=8, max_gen_len=None):
    check_gpu_memory()
    generator = Llama.build(ckpt_dir=ckpt_dir, tokenizer_path=tokenizer_path, max_seq_len=max_seq_len, max_batch_size=max_batch_size)

    dialogs = deque([], maxlen=4)  # Initialize deque with maximum length of 4
    last_input = ""

    try:
        while True:
            current_input = input("Enter your instruction (type 'all done' to finish): ")
            if current_input.lower() == "all done":
                break
            if current_input == "":
                print('Empty instruction, please try again.')
                continue

            start_time = time.time()
            if last_input != "" and similarity(last_input, current_input):
                combined_content = last_input + " " + current_input
                dialogs.append([{"role": "user", "content": combined_content}])
            else:
                dialogs.append([{"role": "user", "content": current_input}])

            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"Comparision Similar Time: {elapsed_time:.2f} seconds")

            last_input = current_input

            start_time = time.time()
            results = generator.chat_completion([dialogs[-1]], max_gen_len=max_gen_len, temperature=temperature, top_p=top_p)
            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"Inference Elapsed time: {elapsed_time:.2f} seconds")

            for dialog, result in zip([dialogs[-1]], results):
                for msg in dialog:
                    print(f"{msg['role'].capitalize()}: {msg['content']}\n")
                print(f"> {result['generation']['role'].capitalize()}: {result['generation']['content']}")
                print("\n==================================\n")

    except KeyboardInterrupt:
        print("Interrupted by user.")

    print("Session Ended.")

if __name__ == "__main__":
    fire.Fire(main)
