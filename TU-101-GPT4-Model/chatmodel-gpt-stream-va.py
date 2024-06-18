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
def ask_openai_for_service_extraction(question, api_key, conversation_history, output_placeholder):
    combined_query = f"{question}"
    full_conversation = conversation_history + [{"role": "user", "content": combined_query}]

    response = client.chat.completions.create(
        model="gpt-4",
        messages=full_conversation,
        stream=True,
    )

    result = ""
    for chunk in response:
        if chunk.choices[0].delta.content is not None:
            result += chunk.choices[0].delta.content
            output_placeholder.text_area("Answer", result, height=300)

    if result != "":
        conversation_history.append({"role": "user", "content": combined_query})
        conversation_history.append({"role": "assistant", "content": result})
    return result

# Streamlit UI
st.markdown("# PPD Chatbot")
st.markdown("### Ask me about available services:")
user_query = st.text_area("Enter your query (e.g., 'What is PPD')", key="user_query")

# Checkbox to combine query with URL
combine_query_with_url = st.checkbox("Reference websites in the query")

# Conditionally display the URL text field
url_text = ""
if combine_query_with_url:
    url_text = st.text_area("Enter URL text", key="url_text")

# Submit button
submit_button = st.button("Submit")

# Initialize global variables
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []

# Create a placeholder for the output text area
output_placeholder = st.empty()

if submit_button and user_query:
    start_time = time.time()

    if combine_query_with_url and url_text:
        user_query += f"The answer please reference the website: {url_text}"

    async def main():
        ask_openai_for_service_extraction(user_query, api_key, st.session_state.conversation_history, output_placeholder)
        end_time = time.time()
        elapsed_time = end_time - start_time
        #st.write(f"Inference Elapsed time: {elapsed_time:.2f} seconds")

    asyncio.run(main())
