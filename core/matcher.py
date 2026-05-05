# matcher.py
import os

import numpy as np
from numpy.linalg import norm


def cosine_similarity(a, b):
    return np.dot(a, b) / (norm(a) * norm(b))


def find_person(new_emb, database, threshold=0.5):
    best_person = None
    best_score = 0

    for person, data in database.items():
        embeddings = data["embeddings"]
        mean_emb = data["mean"]

        score = max(cosine_similarity(new_emb, e) for e in embeddings)
        mean_score = cosine_similarity(new_emb, mean_emb)

        if score > best_score and score > threshold and mean_score > threshold:
            best_score = score
            best_person = person

    return best_person


def is_certain_match(person, face_embedding, threshold=0.5):

    embeddings = []
    scores = []
    folder = "./dataset/embeddings/" + person

    for file in os.listdir(folder):
        if file.endswith(".npy"):
            emb = np.load(os.path.join(folder, file))
            embeddings.append(emb)

    for emb in embeddings:
        sim = cosine_similarity(face_embedding, emb)
        scores.append(sim)

    best_score = max(scores)
    print("Best similarity:", best_score)

    return best_score > threshold
