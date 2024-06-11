# -*- coding: utf-8 -*-
from collections import deque
import fire
import time
import torch
from sympy import flatten


from llama import Llama, Dialog
#from langchain.prompts import Conversation, Memory
#from langchain.schema import PromptPart




def respond_to_user(input_text,conversationSet,modelset):
    # Update the conversation with the user's latest input
    conversationSet.update_user_input(input_text)

    # Generate a prompt using the conversation history and memory
    prompt = conversationSet.get_prompt(parts=[
        PromptPart.SYSTEM_MEMORY,
        PromptPart.USER_INPUT
    ])

    # Use the LLaMA model to generate a response
    response = modelset.generate(prompt)

    # Update the conversation with the generated response
    conversationSet.update_system_response(response)

    # Return the response to the user
    return response


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


class ConversationMemory:
    def __init__(self):
        self.history = deque([], maxlen=4)  # Initialize deque with maximum length of 4

    def update_user_input(self, input_text):
        self.history.append({"role": "user", "content": input_text})

    def update_assistant_response(self, response_text):
        self.history.append({"role": "assistant", "content": response_text})

    def get_conversation_prompt(self):
        # Create a prompt by joining all conversation parts
        prompt = " ".join([item['content'] for item in self.history])
        return prompt

    def display_history(self):
        # Utility function to display current conversation memory for debugging
        print("Current Conversation Memory:")
        for item in self.history:
            print(f"{item['role']}: {item['content']}")



def main(ckpt_dir, tokenizer_path, temperature=0.6, top_p=0.9, max_seq_len=512, max_batch_size=8, max_gen_len=None):
    check_gpu_memory()

    # Initialize the LLaMA model
    generator = Llama.build(ckpt_dir=ckpt_dir, tokenizer_path=tokenizer_path, max_seq_len=max_seq_len, max_batch_size=max_batch_size)

    #last_input = ""

    # Initialize the conversation memory
    conversation = ConversationMemory()

    try:
        while True:
            dialogs = []
            print("\n**************************************\n")
            current_input = input("Enter your instruction (type 'exit' to finish): ")



            if current_input.lower() == "exit":
                break
            if current_input == "":
                print('Empty instruction, please try again.')
                continue

            '''
            start_time = time.time()
            if last_input != "" and similarity(last_input, current_input):
                combined_content = last_input + " " + current_input
                dialogs.append([{"role": "user", "content": combined_content}])
            else:
                dialogs.append([{"role": "user", "content": current_input}])

            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"Comparision Similar Time: {elapsed_time:.2f} seconds")
            '''

            dialogs.append([{"role": "user", "content": current_input}])


            #last_input = current_input

            # Update the conversation with the user's latest input
            conversation.update_user_input(current_input)

            # Generate a prompt using the conversation history and memory
            prompt_input = conversation.get_conversation_prompt()

            print(f" Conversation Prompt: {prompt_input}")
            print(f" Prompt: {len(prompt_input)}")

            #dialogs.append([{"role": "user", "content": prompt_input}])

            start_time = time.time()

            results = generator.chat_completion([dialogs[-1]], max_gen_len=max_gen_len, temperature=temperature, top_p=top_p)
            #results = generator.chat_completion(prompt, max_gen_len=max_gen_len, temperature=temperature, top_p=top_p)


            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"Inference Elapsed time: {elapsed_time:.2f} seconds")


            # Update the conversation with the generated response
            print('Results:',results)
            print('Results[0]:',results[-1]['generation']['content'])

            # Update the conversation with the generated response
            conversation.update_assistant_response(results[0]['generation']['content'])
            dialogs.append([{"role": "assistant", "content": results[0]['generation']['content']}])

            print(dialogs[-1])

            # Display the conversation history
            print("\n==================================\n")
            print(f"Conversation Memory: {conversation.display_history}")
            print(f"Conversation Memory: {conversation.get_conversation_prompt()}")
            print("\n==================================\n")

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
