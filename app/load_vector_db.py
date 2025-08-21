import json
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import os
from dotenv import load_dotenv

load_dotenv()

project_path = os.getcwd()
documents_path = os.path.join(project_path, 'data_documents')
vector_stores_path = os.path.join(project_path, 'data_vector_stores')


chunks_paths = []

for project_dir in os.listdir(documents_path):
    chunks_path = os.path.join(documents_path, project_dir, 'chunks.json')
    chunks_paths.append(chunks_path)

chunks = []

for path in chunks_paths:
    with open(path, 'r', encoding='utf-8') as file:
        chunks.extend(json.load(file))

documents = [
    Document(page_content=chunk['page_content'], metadata=chunk['metadata']) 
    for chunk in chunks
]

embedding_model = GoogleGenerativeAIEmbeddings(model='models/gemini-embedding-001')

vector_db = FAISS.from_documents(documents, embedding_model, distance_strategy='COSINE')
vector_db.save_local(vector_stores_path)



