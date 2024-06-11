

from collections import deque

class ConversationMemory:
    def __init__(self):
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

# Initialize the conversation memory
conversation = ConversationMemory()

# Update with user input
conversation.update_user_input("I am going to Paris, what should I see?")

# Update with assistant response
conversation.update_assistant_response("""\
Paris, the capital of France, is known for its stunning architecture, art museums, historical landmarks, and romantic atmosphere. Here are some of the top attractions to see in Paris:

1. The Eiffel Tower: The iconic Eiffel Tower is one of the most recognizable landmarks in the world and offers breathtaking views of the city.
2. The Louvre Museum: The Louvre is one of the world's largest and most famous museums, housing an impressive collection of art and artifacts, including the Mona Lisa.
3. Notre-Dame Cathedral: This beautiful cathedral is one of the most famous landmarks in Paris and is known for its Gothic architecture and stunning stained glass windows.

These are just a few of the many attractions that Paris has to offer. With so much to see and do, it's no wonder that Paris is one of the most popular tourist destinations in the world.""")

# Update with another user input
conversation.update_user_input("What is so great about #1?")

# Display the formatted history
formatted_history = conversation.get_history()
print(formatted_history)
