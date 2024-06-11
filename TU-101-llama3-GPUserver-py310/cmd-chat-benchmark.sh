#!/bin/bash
torchrun --nproc_per_node 1 example_chat_completion_bench.py     --ckpt_dir Meta-Llama-3-8B-Instruct/     --tokenizer_path Meta-Llama-3-8B-Instruct/tokenizer.model   >> out-chat.txt
