#import openai
from dotenv import load_dotenv
import os
import time
from openai import OpenAI

# Load .env file
load_dotenv()
# Retrieve API key
api_key = os.getenv("API_KEY")

# Initialize OpenAI client
#openai.api_key = api_key
api_key = os.getenv("API_KEY")

client = OpenAI(api_key=api_key)

# Function to query OpenAI API for extracting the service and zipcode
def ask_openai_for_service_extraction(question, conversation_history):
    combined_query = f"{question}"
    full_conversation = conversation_history + [{"role": "user", "content": combined_query}]

    #try:
    response = client.chat.completions.create(model="gpt-4", messages=full_conversation)

    #response = openai.ChatCompletion.create(
    #    model="gpt-4",
    #    messages=full_conversation
    #)

    if response.choices:
        #conversation_history.append({"role": "user", "content": combined_query})
        #conversation_history.append({"role": "assistant", "content": response.choices[0].message['content']})

        conversation_history.append({"role": "user", "content": combined_query})
        conversation_history.append({"role": "assistant", "content": response.choices[0].message.content})

        #return response.choices[0].message['content']
        return response.choices[0].message.content

    #except client.error.OpenAIError as e:
    #except client.exceptions.OpenAIError as e:
    #    print(f"Error: {e}")
    #    return None


def main():
    conversation_history = []

    while True:
        user_query = input("Enter your query (e.g., 'What is PPD') or 'exit' to quit: ")
        if user_query.lower() == 'exit':
            break

        start_time = time.time()
        response = ask_openai_for_service_extraction(user_query, conversation_history)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Inference Elapsed time: {elapsed_time:.2f} seconds")

        if response:
            #extracted_info = response.choices[0].message.content.strip()
            print("Answer:", response)
        else:
            print("No response from API.")


if __name__ == "__main__":
    main()
