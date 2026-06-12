import os
import numpy as np

from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

documents=[]

def load_documents():
    for filename in os.listdir("docs"):
        filepath = os.path.join("docs", filename)
        with open(filepath, "r") as file:
            content = file.read()
            documents.append({ "filename": filename, "content":content})
    return documents

def embed_documents(documents):
    for doc in documents:
        embedding = model.encode(doc["content"])
        doc["embedding"]=embedding
    return documents

def cosine_similarity(v1,v2):
    return np.dot(v1,v2)/(np.linalg.norm(v1)*np.linalg.norm(v2))

def search(query, documents):
    embeded_query = model.encode(query)
    results = []
    for doc in documents:
        score = cosine_similarity(embeded_query, doc["embedding"])
        results.append((score, doc))
    results.sort(reverse=True, key=lambda x:x[0])
    return results

def chunk_documents(documents):
    chunks=[]
    for doc in documents:
        paragraphs = doc["content"].split("\n")
        for sentence in paragraphs:
            if not sentence.strip():
                continue
            chunks.append({"filename":doc["filename"], "chunk":sentence})
    return chunks
        

documents = load_documents()
documents = embed_documents(documents)

chunks = chunk_documents(documents)
print(chunks)


