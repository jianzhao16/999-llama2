import asyncio
from jina import Client
from docarray import BaseDoc
from jina import Executor, requests
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_name = 'Base-Llama-2-7b-chat-hf'

class TokenStreamingExecutor(Executor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, device_map='auto', load_in_8bit=True
        )

class PromptDocument(BaseDoc):
    prompt: str
    max_tokens: int

class ModelOutputDocument(BaseDoc):
    token_id: int
    generated_text: str

'''
from jina import Deployment

with Deployment(uses=TokenStreamingExecutor, port=12345, protocol='grpc') as dep:
    dep.block()

'''

class TokenStreamingExecutor(Executor):
    ...

    def starts_with_space(self, token_id):
        token = self.tokenizer.convert_ids_to_tokens(token_id)
        return token.startswith('▁')

    @requests(on='/stream')
    async def task(self, doc: PromptDocument, **kwargs) -> ModelOutputDocument:
        input = self.tokenizer(doc.prompt, return_tensors='pt')
        input_len = input['input_ids'].shape[1]

        for output_length in range(doc.max_tokens):
            output = self.model.generate(**input, max_new_tokens=1)
            current_token_id = output[0][-1]
            if current_token_id == self.tokenizer.eos_token_id:
                break

            current_token = self.tokenizer.decode(
                current_token_id, skip_special_tokens=True
            )
            if self.starts_with_space(current_token_id.item()) and output_length > 1:
                current_token = ' ' + current_token
            yield ModelOutputDocument(
                token_id=current_token_id,
                generated_text=current_token,
            )

            input = {
                'input_ids': output,
                'attention_mask': torch.ones(1, len(output[0])),
            }



llama_prompt = PromptDocument(
    prompt="""
<s>[INST] <<SYS>>
You are a helpful, respectful, and honest assistant. Always answer as helpfully 
as possible while being safe.  Your answers should not include any harmful, 
unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure 
that your responses are socially unbiased and positive in nature.

If a question does not make any sense or is not factually coherent, explain why 
instead of answering something incorrectly. If you don't know the answer to a 
question, don't share false information.
<</SYS>>
If I punch myself in the face and it hurts, am I weak or strong? [/INST]
""",
    max_tokens=100,
)

async def main():
    client = Client(port=12345, protocol='grpc', asyncio=True)
    async for doc in client.stream_doc(
        on='/stream',
			  inputs=llama_prompt,
			  return_type=ModelOutputDocument,
    ):
        print(doc.generated_text, end='')

asyncio.run(main())