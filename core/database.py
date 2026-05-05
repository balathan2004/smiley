# database.py
import os
import numpy as np


def load_database(root):
    database = {}

    for person in os.listdir(root):
        person_path = os.path.join(root, person)
        if not os.path.isdir(person_path):
            continue

        embeddings = []
        for file in os.listdir(person_path):
            if file.endswith(".npy"):
                emb = np.load(os.path.join(person_path, file))
                embeddings.append(emb)

        if embeddings:
            database[person] = {
                "embeddings": embeddings,
                "mean": np.mean(embeddings, axis=0),
            }

    return database


def load_person_embeddings(folder):
    embeddings = []
    for file in os.listdir(folder):
        if file.endswith(".npy"):
            emb = np.load(os.path.join(folder, file))
            embeddings.append(emb)

    if len(embeddings) == 0:
        return None, None

    mean_emb = np.mean(embeddings, axis=0)
    return embeddings, mean_emb
