"""
PyMuPDF (fitz): Widely considered the fastest open-source library for PDF extraction. It preserves more document structure than older tools like PyPDF2.
python-docx: The industry standard for reading and creating .docx files programmatically.
PyMuPDF4LLM: A specialized tool that converts PDFs directly into clean Markdown or JSON, which is perfect for feeding data into your retraining script.


# Implementation Guide
Step 1: Extracting from PDF
Use this script to pull text from your PDF files. 
"""

import fitz  # PyMuPDF is imported as fitz

def extract_pdf_text(path):
    with fitz.open(path) as doc:
        text = ""
        for page in doc:
            text += page.get_text()
    return text

raw_pdf_content = extract_pdf_text("your_file.pdf")

"""
Step 2: Extracting from Word (.docx)
Use this script for your Word documents. 
"""
from docx import Document

def extract_docx_text(path):
    doc = Document(path)
    # Joins all paragraphs with a newline
    return "\n".join([para.text for para in doc.paragraphs])

raw_word_content = extract_docx_text("your_file.docx")

"""
Step 3: Save to JSONL for Retraining
Once you have the text from both, you can combine them into the training format we discussed earlier.
"""

import json

data_pair = {
    "instruction": "Summarize the following document.",
    "context": "Source: your_file.pdf",
    "response": raw_pdf_content[:1000]  # First 1000 characters as a sample
}

with open("train_data.jsonl", "a") as f:
    f.write(json.dumps(data_pair) + "\n")
