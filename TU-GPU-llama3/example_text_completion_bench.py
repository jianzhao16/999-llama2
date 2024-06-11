import fire
import time
import torch
from llama import Llama
from collections import deque

def similarity(input1, input2):
    """Check if two strings are similar based on common words."""
    words1 = set(input1.split())
    words2 = set(input2.split())
    common = words1.intersection(words2)
    return len(common) > len(words1) / 2  # Example threshold, adjust as needed


def main(ckpt_dir, tokenizer_path, temperature=0.6, top_p=0.9, max_seq_len=128, max_gen_len=64, max_batch_size=4):
    """
    Run the Llama3 model with dynamically modified instructions based on user input.
    Continues until 'all done' is entered.
    """
    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )

    inputs =  deque([], maxlen=2)  # Initialize deque with maximum length of 1

    try:
        while True:
            user_input = input("Enter your instruction (type 'all done' to finish): ")
            if user_input.lower() == "all done":
                break

            # Store inputs and check for similarity with the last two inputs
            if len(inputs) >= 2 and similarity(inputs[-1], user_input) and similarity(inputs[-2], user_input):
                # Combine the last two inputs and the new one
                combined_input = inputs[-2] + " " + inputs[-1] + " " + user_input
                print(f"Combining inputs due to similarity: {combined_input}")
                inputs.append(combined_input)
            else:
                inputs.append(user_input)

            # Generate response using the latest input or combined input
            #prompts = [inputs[-1]]
            prompts = user_input
            start_time = time.time()
            results = generator.text_completion(
                prompts,
                max_gen_len=max_gen_len,
                temperature=temperature,
                top_p=top_p,
            )
            print(f"Answer: > {results[0]['generation']}")
            print("\n==================================\n")
            print(f"Inference Elapsed time: {time.time() - start_time:.2f} seconds")

    except KeyboardInterrupt:
        print("Session interrupted by user.")

    print("Session Ended.")


if __name__ == "__main__":
    fire.Fire(main)
