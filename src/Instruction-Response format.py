"""
How to Load it in Python
You can use the datasets library to load this file instantly for your QLoRA script.
python
"""


from datasets import load_dataset

# Load your local JSONL file
dataset = load_dataset('json', data_files='train_data.jsonl', split='train')

# Format the data into a single string for the LLM
def format_prompts(example):
    text = f"### Instruction:\n{example['instruction']}\n\n### Context:\n{example['context']}\n\n### Response:\n{example['response']}"
    return {"text": text}

dataset = dataset.map(format_prompts)
print(dataset[0]['text']) # Verify the first example

