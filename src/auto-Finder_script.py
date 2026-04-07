"""
The "Auto-Finder" Script
This script will look through all folders (and subfolders) from a starting point you choose (like your Documents folder) and extract the text automatically.
"""

import os
import fitz  # PyMuPDF
from docx import Document
import json

def get_text_from_any_file(file_path):
    try:
        if file_path.endswith('.pdf'):
            with fitz.open(file_path) as doc:
                return "".join([page.get_text() for page in doc])
        elif file_path.endswith('.docx'):
            doc = Document(file_path)
            return "\n".join([p.text for p in doc.paragraphs])
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
    return None

# --- Configuration ---
search_directory = "C:/Users/YourName/Documents"  # Change this to your main folder
output_file = "raw_extracted_data.jsonl"

# --- The Search & Extract Loop ---
with open(output_file, "w", encoding="utf-8") as f:
    for root, dirs, files in os.walk(search_directory):
        for file in files:
            if file.endswith(('.pdf', '.docx')):
                full_path = os.path.join(root, file)
                print(f"Processing: {file}")
                
                content = get_text_from_any_file(full_path)
                if content and len(content.strip()) > 50: # Ignore empty/tiny files
                    # Structure it for your training
                    entry = {
                        "instruction": "Explain the details from this document.",
                        "context": f"Source: {file}",
                        "response": content[:2000] # Taking first 2000 chars as a sample
                    }
                    f.write(json.dumps(entry) + "\n")

print(f"\nDone! All text is now in {output_file}")

