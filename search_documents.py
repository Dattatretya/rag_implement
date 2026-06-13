import os
import numpy as np
import ollama
import json

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

# def embed_documents(documents):
#     for doc in documents:
#         embedding = model.encode(doc["content"])
#         doc["embedding"]=embedding
#     return documents

def cosine_similarity(v1,v2):
    return np.dot(v1,v2)/(np.linalg.norm(v1)*np.linalg.norm(v2))

def chunk_documents(documents):
    chunks=[]
    for doc in documents:
        paragraphs = doc["content"].split("\n")
        for sentence in paragraphs:
            if not sentence.strip():
                continue
            chunks.append({"filename":doc["filename"], "chunk":sentence})
    return chunks

def embed_chunks(chunks):
    for chunk in chunks:
        embedded_chunk = model.encode(chunk["chunk"])
        chunk["embedding"]=embedded_chunk
    return chunks

def save_chunks(chunks):
    serialize_chunk = []
    for chunk in chunks:
        serialize_chunk.append({"filename":chunk["filename"], "chunk":chunk["chunk"], "embedding":chunk["embedding"].tolist()})
    with open("vector_store.json", "w") as file:
        json.dump(serialize_chunk, file, indent=4)


def search(query, chunks):
    embeded_query = model.encode(query)
    results = []
    for chunk in chunks:
        score = cosine_similarity(embeded_query, chunk["embedding"])
        results.append((score, chunk))
    results.sort(reverse=True, key=lambda x:x[0])
    return results

def build_context(results):
    top_chunks= results[:3]
    context = ""
    for score, chunk in top_chunks:
        context+=chunk["chunk"]+"\n\n"
    return context

def build_prompt(context, query):
    prompt = f"""
            Answer the question using ONLY the context below.

            Context:
            {context}

            Question:
            {query}

            Answer:
            """
    return prompt

def ask_llm(prompt):
    response=ollama.chat(
        model="mistral",
        messages=[{
            "role":"user",
            "content":prompt
        }]
    )
    return response["message"]["content"]
        

documents = load_documents()
# documents = embed_documents(documents)

chunks = chunk_documents(documents)
chunks=embed_chunks(chunks)

query = input("Enter your question  ")

results = search(query, chunks)
context = build_context(results)

prompt=build_prompt(context, query)

answer = ask_llm(prompt)

save_chunks(chunks)
print(answer)

