# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import fire
import torch

from llama import Llama
from typing import List

import sys
import io

import time

#sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def main(
        ckpt_dir: str,
        tokenizer_path: str,
        temperature: float = 0.6,
        top_p: float = 0.9,
        max_seq_len: int = 256,
        max_gen_len: int = 64,
        max_batch_size: int = 4,
):
    """
    Entry point of the program for generating text using a pretrained model.

    Args:
        ckpt_dir (str): The directory containing checkpoint files for the pretrained model.
        tokenizer_path (str): The path to the tokenizer model used for text encoding/decoding.
        temperature (float, optional): The temperature value for controlling randomness in generation.
            Defaults to 0.6.
        top_p (float, optional): The top-p sampling parameter for controlling diversity in generation.
            Defaults to 0.9.
        max_seq_len (int, optional): The maximum sequence length for input prompts. Defaults to 128.
        max_gen_len (int, optional): The maximum length of generated sequences. Defaults to 64.
        max_batch_size (int, optional): The maximum batch size for generating sequences. Defaults to 4.
    """

        # Start the timer
    start_time = time.time()

    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )

    # Calculate the elapsed time
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Load Model Elapsed time: {elapsed_time:.2f} seconds")


    start_time = time.time()

    prompts: List[str] = [
        # For these prompts, the expected answer is the natural continuation of the prompt
        #"I believe the meaning of life",
	#"How are you today?",
	#"Where is good place visit in Philly",
        #"Simply put, the theory of relativity states that ",
        # Few shot prompt (providing a few examples before asking model to complete more);
        #""Translate English to French:
        #sea otter => loutre de mer
        #peppermint => menthe poivrée
        #plush girafe => girafe peluche
        #cheese =>""",

	#""Extract the type of service and zipcode from the following user query:
        #I want to food service nearby 19140 => service:Food, zipcode:19140, status:okay
        #He want to food lunch nearby 19132 => service:Food, zipcode:19132, status:okay
        #Counseling services => service: Mental Health, zipcode:000000, status:okay
        #Temporary housing => service:Shelter,zipcode:000000, status:okay
	#He want to find Mental Health service nearby 19123 =>""",

        """Extract service and zipcode:
        I want food service nearby 19140 => service:Food, zip:19140, status:okay
        Counseling services => service: Mental Health, zip:000000, status:okay
        Temporary housing => service:Shelter, zip:000000, status:okay
        He want to find Mental Health service nearby 19123 =>""",
      
    ]

    #import time



    # Your code here
    # Example:
    #for i in range(1000000):
    #   pass

    # End the timer
    #end_time = time.time()

    # Calculate the elapsed time
    #elapsed_time = end_time - start_time

    #print(f"Elapsed time: {elapsed_time:.2f} seconds")

    results = generator.text_completion(
        prompts,
        max_gen_len=max_gen_len,
        temperature=temperature,
        top_p=top_p,
    )
    for prompt, result in zip(prompts, results):
        print('Question:\n'+prompt+'\n')
        print(f"Answer: > {result['generation']}")
        print("\n==================================\n")

    # Calculate the elapsed time
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Inference Elapsed time: {elapsed_time:.2f} seconds")


if __name__ == "__main__":
    #torch.set_default_dtype(torch.float16)  # Equivalent to torch.FloatTensor data type

    #torch.set_default_device('cuda')  # Set default tensor allocation on GPU

    #sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

    fire.Fire(main)
