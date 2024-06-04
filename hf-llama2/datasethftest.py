import pandas as pd
from datasets import load_dataset, Dataset
from huggingface_hub import login

# Initialize Hugging Face authentication
hf_auth = "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"
#login(token=hf_auth)
login(token=hf_auth,add_to_git_credential=True)

#dataset = load_dataset("BI55/MedText", split="train")
file_path = './ourdata-ppd.csv'
df = pd.read_csv(file_path)
print(df)

#df = pd.DataFrame(dataset)
prompt = df.pop("Prompt")
comp = df.pop("Completion")
df["Info"] = prompt + "\n" + comp
print(df["Info"])
