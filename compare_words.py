from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer("all-MiniLM-L6-v2")

def cosine_similarity(v1,v2):
    return np.dot(v1,v2)/(np.linalg.norm(v1)*np.linalg.norm(v2))

pairs = [
    ("vacation", "holiday"),
    ("doctor", "nurse"),
    ("python", "programming"),
    ("dog", "cat"),
    ("vacation", "airplane"),
]

for word1, word2 in pairs:
    vec1=model.encode(word1)
    vec2=model.encode(word2)

    score = cosine_similarity(vec1, vec2)

    print(f"{word1} <-> {word2}: {score:.4f}")