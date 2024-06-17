import streamlit as st
from openai import OpenAI

from dotenv import load_dotenv
import os
# Load .env file
load_dotenv()
# Retrieve API key
api_key = os.getenv("API_KEY")

client = OpenAI(api_key=api_key)
import os
from dotenv import load_dotenv
import time


# Ensure required packages are installed
# pip install openai==1.0.0 streamlit-folium folium pandas python-dotenv pgeocode

# Function to query OpenAI API for extracting the service and zipcode
def ask_openai_for_service_extraction(question, api_key, conversation_history):
    combined_query = f"{question}"
    full_conversation = conversation_history + [{"role": "user", "content": combined_query}]

    response = client.chat.completions.create(model="gpt-4", messages=full_conversation)

    if response.choices:
        conversation_history.append({"role": "user", "content": combined_query})
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
    response = ask_openai_for_service_extraction(user_query, api_key, st.session_state.conversation_history)
    end_time = time.time()
    elapsed_time = end_time - start_time
    st.write(f"Inference Elapsed time: {elapsed_time:.2f} seconds")

    if response.choices:
        extracted_info = response.choices[0].message.content.strip()
        st.write("Answer:", extracted_info)
