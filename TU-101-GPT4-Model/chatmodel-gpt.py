import streamlit as st

from openai import OpenAI

from dotenv import load_dotenv
import os


# Load .env file
load_dotenv()
# Retrieve API key
api_key = os.getenv("API_KEY")

client = OpenAI(api_key=api_key)
from branca.element import IFrame

import os
from dotenv import load_dotenv
import time

#pip install openai==0.28
#pip install streamlit-folium
#pip install folium
#pip install pandas
#pip install python-dotenv
#pip install pgeocode


# Function to query OpenAI API for extracting the service and zipcode
def ask_openai_for_service_extraction(question, api_key, conversation_history):
    extraction_instruction = "Extract the type of service and zipcode from the following user query:"
    #extraction_instruction =""
    #combined_query = f"{extraction_instruction}\n{question}"
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
#user_query = st.text_input("Enter your query (e.g., 'What is PPD')", key="user_query")
user_query = st.text_area("Enter your query (e.g., 'What is PPD')", key="user_query")

# Submit button
submit_button = st.button("Submit")

# Initialize global variables
conversation_history = []



#api_key = ''  # Replace this with your actual OpenAI API key

if submit_button:

    start_time = time.time()
    response = ask_openai_for_service_extraction(user_query, api_key, conversation_history)
    # print time elapsed
    end_time = time.time()
    elapsed_time = end_time - start_time
    st.write(f"Inference Elapsed time: {elapsed_time:.2f} seconds")

    if response.choices:
        extracted_info = response.choices[0].message.content.strip()

        # Debugging: Display extracted information
        st.write("Answer:", extracted_info)

