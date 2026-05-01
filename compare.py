import numpy as np
from numpy.linalg import norm
import os
import cv2
from insightface.app import FaceAnalysis


app = FaceAnalysis(providers=['CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))

# 📸 Load new image
test_img = cv2.imread("./faces/test/1.jpg")

if test_img is None:
    print("Test image not found ❌")
    exit()

faces = app.get(test_img)

if len(faces) == 0:
    print("No face found ❌")
    exit()

# 👉 This is your missing piece
new_emb = faces[0].embedding



embeddings = []
folder = "./dataset/embeddings/person"

def cosine_similarity(a, b):
    return np.dot(a, b) / (norm(a) * norm(b))


for file in os.listdir(folder):
    if file.endswith(".npy"):
        emb = np.load(os.path.join(folder, file))
        embeddings.append(emb)


# 🔍 Compare with new embedding
def is_same_person(new_emb, threshold=0.5):
    scores = []

    for emb in embeddings:
        sim = cosine_similarity(new_emb, emb)
        scores.append(sim)

    best_score = max(scores)
    print("Best similarity:", best_score)

    return best_score > threshold    

mean_emb = np.mean(embeddings, axis=0)

best_score = max([cosine_similarity(new_emb, e) for e in embeddings])
mean_score = cosine_similarity(new_emb, mean_emb)

print("Best:", best_score)
print("Mean:", mean_score)

if best_score > 0.5 and mean_score > 0.45:
    print("Same person ✅")
else:
    print("Different person ❌")
