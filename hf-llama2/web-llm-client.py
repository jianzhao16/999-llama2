import requests

def get_generated_text(prompt, max_new_tokens=128):
    url = "http://localhost:8000/generate-text"
    payload = {"text": prompt, "max_new_tokens": max_new_tokens}
    response = requests.post(url, json=payload)
    if response.status_code == 200:
        return response.json()["generated_text"]
    else:
        return f"Error: {response.status_code}, {response.text}"

if __name__ == "__main__":
    while True:
        user_input = input("Enter your instruction (type 'exit' to finish): ")
        if user_input.lower() == "exit":
            break
        if user_input.strip() == "":
            print('Empty instruction, please try again.')
            continue
        generated_text = get_generated_text(user_input)
        print(f"Generated Text: {generated_text}")
