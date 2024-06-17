import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv
import os
import time
import asyncio

# Load .env file
load_dotenv()

# Retrieve API key
api_key = os.getenv("API_KEY")

# Initialize OpenAI client
client = OpenAI(api_key=api_key)

# Function to query OpenAI API for extracting the service and zipcode
async def ask_openai_for_service_extraction(question, api_key, conversation_history):
    combined_query = f"{question}"
    full_conversation = conversation_history + [{"role": "user", "content": combined_query}]

    response = client.chat.completions.create(
        model="gpt-4",
        messages=full_conversation,
        stream=True,
    )


    for chunk in response:
        if chunk.choices[0].delta.content is not None:
            print(chunk.choices[0].delta.content, end="")


    if response.choices:
        conversation_history.append({"role": "user", "content": combined_query})
        #conversation_history.append({"role": "assistant", "content": response.choices[0].message['content']})
        conversation_history.append({"role": "assistant", "content": response.choices[0].message.content})
    return response

# Streamlit UI
st.markdown("# PPD Chatbot")
st.markdown("### Ask me about available services:")
user_query = st.text_area("Enter your query (e.g., 'What is PPD')", key="user_query")

# Submit button
submit_button = st.button("Submit")

# Initialize global variables
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []

if submit_button and user_query:
    start_time = time.time()

    #async def main():

    #response = await (user_query, api_key, st.session_state.conversation_history)
    conversation_history = st.session_state.conversation_history
    combined_query = f"{user_query}"
    full_conversation = conversation_history + [{"role": "user", "content": combined_query}]

    stream = client.chat.completions.create(
        model="gpt-4",
        messages=full_conversation,
        stream=True,
    )

    for chunk in stream:
        if 'choices' in chunk:
            for choice in chunk['choices']:
                if 'delta' in choice:
                    st.write(chunk.choices[0].delta.content, end='', flush=True)

        if chunk.choices[0].message.content is not None:
            conversation_history.append({"role": "user", "content": combined_query})
            # conversation_history.append({"role": "assistant", "content": response.choices[0].message['content']})
            conversation_history.append({"role": "assistant", "content": chunk.choices[0].message.content})

    end_time = time.time()
    elapsed_time = end_time - start_time
    st.write(f"Inference Elapsed time: {elapsed_time:.2f} seconds")

    '''
        if response.choices:
            #extracted_info = response.choices[0].message['content'].strip()
            extracted_info = response.choices[0].message.content.strip()
            st.write("Answer:", extracted_info)
    '''
    #asyncio.run(main())
