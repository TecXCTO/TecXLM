"""
The Auto-Chunking Script
This script finds your documents, extracts the text, and breaks it into chunks of 800 words with a 100-word overlap. Overlapping is crucial so the model doesn't lose context between the end of one chunk and the start of the next.

"""
import os
import json
import fitz  # PyMuPDF
from docx import Document

def get_text(file_path):
    try:
        if file_path.endswith('.pdf'):
            with fitz.open(file_path) as doc:
                return " ".join([page.get_text() for page in doc])
        elif file_path.endswith('.docx'):
            doc = Document(file_path)
            return " ".join([p.text for p in doc.paragraphs])
    except Exception: return None

def chunk_text(text, size=800, overlap=100):
    words = text.split()
    return [" ".join(words[i : i + size]) for i in range(0, len(words), size - overlap)]

# Configuration
source_folder = "C:/Your/Folder/Path" 
output_file = "chunked_train_data.jsonl"

with open(output_file, "w", encoding="utf-8") as f:
    for root, _, files in os.walk(source_folder):
        for file in files:
            if file.endswith(('.pdf', '.docx')):
                full_path = os.path.join(root, file)
                raw_text = get_text(full_path)
                
                if raw_text:
                    # Break long text into manageable pieces
                    chunks = chunk_text(raw_text)
                    for i, chunk in enumerate(chunks):
                        entry = {
                            "instruction": "Summarize or explain this document section.",
                            "context": f"File: {file} | Part: {i+1}",
                            "response": chunk
                        }
                        f.write(json.dumps(entry) + "\n")

print(f"Extraction complete! Created {output_file}")
