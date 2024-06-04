import torch
import transformers

from langchain.llms import HuggingFacePipeline

# Set quantization configuration to load large model with less GPU memory
# This requires the `bitsandbytes` library
bnb_config = transformers.BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16
)

model_id = 'meta-llama/Llama-2-7b-hf'
hf_auth = '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'

# Initialize the model and tokenizer
model_config = transformers.AutoConfig.from_pretrained(
    model_id,
    use_auth_token=hf_auth
)
model = transformers.AutoModelForCausalLM.from_pretrained(
    model_id,
    trust_remote_code=True,
    config=model_config,
    quantization_config=bnb_config,
    device_map='auto',
    use_auth_token=hf_auth
)
model.eval()

tokenizer = transformers.AutoTokenizer.from_pretrained(
    model_id,
    use_auth_token=hf_auth
)

# Set up the generation pipeline
generate_text = transformers.pipeline(
    model=model, tokenizer=tokenizer,
    return_full_text=True,
    task='text-generation',
    temperature=0.0,
    max_new_tokens=512,
    repetition_penalty=1.1
)

llm = HuggingFacePipeline(pipeline=generate_text)

from langchain.memory import ConversationBufferWindowMemory
from langchain.agents import load_tools, initialize_agent

# Initialize memory and tools
memory = ConversationBufferWindowMemory(
    memory_key="chat_history", k=5, return_messages=True, output_key="output"
)
tools = load_tools(["llm-math"], llm=llm)

# Initialize agent
agent = initialize_agent(
    agent="chat-conversational-react-description",
    tools=tools,
    llm=llm,
    verbose=True,
    early_stopping_method="generate",
    memory=memory
)

# Example user interactions
examples = [
    "What is the square root of 16?",
    "Can you help me calculate the area of a circle with a radius of 5?",
    "What's 15% of 200?"
]

# Process each example
for example in examples:
    response = agent.process({"user_input": example})
    print(f"User: {example}")
    print(f"Assistant: {response['output']}\n")



# special tokens used by llama 2 chat
B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

# create the system message
sys_msg = "<s>" + B_SYS + """Assistant is a expert JSON builder designed to assist with a wide range of tasks.

Assistant is able to respond to the User and use tools using JSON strings that contain "action" and "action_input" parameters.

All of Assistant's communication is performed using this JSON format.

Assistant can also use tools by responding to the user with tool use instructions in the same "action" and "action_input" JSON format. Tools available to Assistant are:

- "Calculator": Useful for when you need to answer questions about math.
  - To use the calculator tool, Assistant should write like so:
    ```json
    {{"action": "Calculator",
      "action_input": "sqrt(4)"}}
    ```

Here are some previous conversations between the Assistant and User:

User: Hey how are you today?
Assistant: ```json
{{"action": "Final Answer",
 "action_input": "I'm good thanks, how are you?"}}
```
User: I'm great, what is the square root of 4?
Assistant: ```json
{{"action": "Calculator",
 "action_input": "sqrt(4)"}}
```
User: 2.0
Assistant: ```json
{{"action": "Final Answer",
 "action_input": "It looks like the answer is 2!"}}
```
User: Thanks could you tell me what 4 to the power of 2 is?
Assistant: ```json
{{"action": "Calculator",
 "action_input": "4**2"}}
```
User: 16.0
Assistant: ```json
{{"action": "Final Answer",
 "action_input": "It looks like the answer is 16!"}}
```

Here is the latest conversation between Assistant and User.""" + E_SYS
new_prompt = agent.agent.create_prompt(
    system_message=sys_msg,
    tools=tools
)
agent.agent.llm_chain.prompt = new_prompt


instruction = B_INST + " Respond to the following in JSON with 'action' and 'action_input' values " + E_INST
human_msg = instruction + "\nUser: {input}"

agent.agent.llm_chain.prompt.messages[2].prompt.template = human_msg