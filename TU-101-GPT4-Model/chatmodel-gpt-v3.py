import openai

from dotenv import load_dotenv
import time
import os

from dotenv import load_dotenv
# Load .env file
load_dotenv()
# Retrieve API key
# Set up your OpenAI API key

# all client options can be configured just like the `OpenAI` instantiation counterpart
# TODO: The 'openai.base_url' option isn't read in the client API. You will need to pass it when you instantiate the client, e.g. 'OpenAI(base_url="https://...")'
# openai.base_url = "https://..."
# TODO: The 'openai.default_headers' option isn't read in the client API. You will need to pass it when you instantiate the client, e.g. 'OpenAI(default_headers={"x-foo": "true"})'
# openai.default_headers = {"x-foo": "true"}

completion = openai.chat.completions.create(
    model="gpt-4",
    messages=[
        {
            "role": "user",
            "content": "How do I output all files in a directory using Python?",
        },
    ],
)
print(completion.choices[0].message.content)