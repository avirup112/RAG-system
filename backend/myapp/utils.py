import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
import os 
from docx import Document as DocxDocument
import fitz

model = SentenceTransformer('all-MiniLM-L6-v2')

def save_faiss_index(index, doc_id):
    faiss.write_index(index, f"faiss_indexes/{doc_id}.index")
    
def load_faiss_index(doc_id):
    return faiss.read_index(f"faiss_indexes/{doc_id}.index")

def get_embeddings(text_list):
    return model.encode(text_list)

def extract_text_from_file(file):
    if file.name.endswith('.txt'):
        return file.read().decode("utf-8")
    elif file.name.endswith('.pdf'):
        doc = DocxDocument(file)
        return '\n'.join([para.text for para in doc.paragraphs])
    elif file.name.endswith('.pdf'):
        text = ""
        pdf = fitz.open(stream=file.read(), filetype="pdf")
        for page in pdf:
            text += page
        return text
    else:
        raise ValueError("Unsupported file type")