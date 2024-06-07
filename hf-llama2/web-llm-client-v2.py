import streamlit as st
import requests

def get_generated_text(prompt, max_new_tokens=128):
    url = "http://localhost:8000/generate-text"
    payload = {"text": prompt, "max_new_tokens": max_new_tokens}
    response = requests.post(url, json=payload)
    if response.status_code == 200:
        return response.json()["generated_text"]
    else:
        return f"Error: {response.status_code}, {response.text}"

# Streamlit app layout
st.title("Text Generation with LLaMA Model")
st.write("Enter your instruction below and click the button to generate text.")

# Input form
user_input = st.text_area("Enter your instruction:", height=150)
max_new_tokens = st.slider("Max New Tokens:", min_value=10, max_value=512, value=128)

if st.button("Generate Text"):
    if user_input.strip() == "":
        st.write("Empty instruction, please try again.")
    else:
        with st.spinner("Generating text..."):
            generated_text = get_generated_text(user_input, max_new_tokens)
        st.write("### Generated Text:")
        st.write(generated_text)
