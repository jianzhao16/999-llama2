#!/bin/bash
torchrun --nproc_per_node 1 example_text_completion.py \                                                                    --ckpt_dir llama-2-7b/ \                                                                                                --tokenizer_path tokenizer.model \
    --max_seq_len 64 --max_batch_size 4 temperature 0.6 >> out-txt.txt

