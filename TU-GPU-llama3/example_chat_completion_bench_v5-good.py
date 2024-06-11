# -*- coding: utf-8 -*-
from collections import deque
import fire
import time
import torch
from sympy import flatten

from llama import Llama, Dialog

# from langchain.prompts import Conversation, Memory
# from langchain.schema import PromptPart


'''

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
'''


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

class ConversationMemoryv1:
    def __init__(self):
        #self.history = deque([], maxlen=4)  # Initialize deque with maximum length of 4
        self.history = deque([], maxlen=4)  # Initialize deque with maximum length of 4

    def update_user_input(self, input_text):
        # Append a dictionary with the user's role and content
        self.history.append({"role": "user", "content": input_text})

    def update_assistant_response(self, response_text):
        # Append a dictionary with the assistant's role and content
        self.history.append({"role": "assistant", "content": response_text})

    def get_history(self):
        # Return a list of all conversation items
        return list(self.history)

    def display_history(self):
        # Print each item in the history
        for item in self.history:
            print(f"{item['role'].capitalize()}: {item['content']}\n")


def main(ckpt_dir, tokenizer_path, temperature=0.6, top_p=0.9, max_seq_len=512, max_batch_size=8, max_gen_len=None):
    #check_gpu_memory()

    # Initialize the LLaMA model
    print("Initializing LLaMA 3 8B model...")
    generator = Llama.build(ckpt_dir=ckpt_dir, tokenizer_path=tokenizer_path, max_seq_len=max_seq_len, max_batch_size=max_batch_size)

    # last_input = ""

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

            print(f" Conversation Prompt: {conversation_input}")
            print(f" Prompt: {conversation.get_total_content_length()}")

            #dialogs.append([{"role": "user", "content": prompt_input}])
            dialogs.append(conversation_input)

            start_time = time.time()

            print('Begin Inference...')
            results = generator.chat_completion([dialogs[-1]], max_gen_len=max_gen_len, temperature=temperature, top_p=top_p)
            #results = generator.chat_completion(conversation_input[-1], max_gen_len=max_gen_len, temperature=temperature, top_p=top_p)
            print('Inference Results:')
            print(results)
            print(" results length:")
            print(len(results))

            #results = [{'generation': {'role': 'assistant', 'content': "I'm happy."}}]

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
            #print(results)
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


if __name__ == "__main__":
    fire.Fire(main)
    #main(ckpt_dir='gpt2', tokenizer_path='gpt2')
