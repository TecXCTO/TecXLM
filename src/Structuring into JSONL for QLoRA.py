"""
Structuring into JSONL for QLoRA
Once you have the text, you need to save it in a line-by-line format where each line is a valid JSON object. This is essential for memory-efficient loading during training. 

Step-by-Step Conversion Script
This script demonstrates how to take a list of extracted text strings and save them into the train_data.jsonl format we discussed.
"""


import jsonlines

# Assume 'raw_texts' is a list of strings extracted from your PDFs/Word docs
raw_texts = [
    "Safety Protocol: Train emergency brakes must be inspected every 24 hours.",
    "Technical Spec: The Snapdragon 73 stabilizer supports up to 160km/h."
]

# Create structured instruction-response pairs
training_data = []
for i, text in enumerate(raw_texts):
    training_data.append({
        "instruction": "Summarize this technical note.",
        "context": f"Document Snippet {i+1}",
        "response": text
    })

# Save as JSONL
with jsonlines.open("train_data.jsonl", mode='w') as writer:
    writer.write_all(training_data)

print("Success: train_data.jsonl is ready for retraining.")

