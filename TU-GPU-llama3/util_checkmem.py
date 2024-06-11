# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from typing import List, Optional

import fire

from llama import Llama, Dialog

import time

import torch

def select_gpu_with_most_free_memory():
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        max_free_memory = 0
        selected_gpu_id = 0

        # Iterate over all GPUs and find the one with the most free memory
        for gpu_id in range(num_gpus):
            torch.cuda.set_device(gpu_id)
            total_memory = torch.cuda.get_device_properties(gpu_id).total_memory
            free_memory = total_memory - torch.cuda.memory_allocated(gpu_id)

            # Update the selected GPU if this one has more free memory
            if free_memory > max_free_memory:
                max_free_memory = free_memory
                selected_gpu_id = gpu_id

        # Set the device to the GPU with the most free memory
        torch.cuda.set_device(selected_gpu_id)
        print(f"Selected GPU {selected_gpu_id} with the most free memory.")
    else:
        print("CUDA is not available. Check your installation or GPU availability.")
def check_gpu_memory():
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        print(f"Number of GPUs available: {num_gpus}")

        for gpu_id in range(num_gpus):
            torch.cuda.set_device(gpu_id)
            total_memory = torch.cuda.get_device_properties(gpu_id).total_memory
            free_memory = total_memory - torch.cuda.memory_allocated(gpu_id)

            # Convert bytes to GB for readability
            total_memory_gb = total_memory / (1024 ** 3)
            free_memory_gb = free_memory / (1024 ** 3)

            print(f"GPU {gpu_id}: Total Memory: {total_memory_gb:.2f} GB, Free Memory: {free_memory_gb:.2f} GB")
    else:
        print("CUDA is not available. Check your installation or GPU availability.")

check_gpu_memory()
