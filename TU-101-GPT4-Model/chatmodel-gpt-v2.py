from flask import Flask, render_template, request, jsonify
from openai import OpenAI

client = OpenAI(api_key=os.getenv("API_KEY"))

import time
import os



from dotenv import load_dotenv
# Load .env file
load_dotenv()
# Retrieve API key
# Set up your OpenAI API key

app = Flask(__name__)

# Function to get response from OpenAI's API
def get_openai_response(prompt):
    response = client.completions.create(engine="text-davinci-003",  # You can use other engines like "gpt-3.5-turbo"
    prompt=prompt,
    max_tokens=150,             # Adjust the response length as needed
    n=1,
    stop=None,
    temperature=0.7             # Adjust the creativity of the response)
    return response.choices[0].text.strip()

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.json.get("message")
    if user_input:
        response = get_openai_response(user_input)
        return jsonify({"response": response})
    return jsonify({"response": "Sorry, I didn't understand that."})

if __name__ == "__main__":
    app.run(debug=True)
