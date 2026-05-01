import cv2
from insightface.app import FaceAnalysis
import numpy as np
from numpy.linalg import norm
import os
import time


app = FaceAnalysis()
app.prepare(ctx_id=0)
cap = cv2.VideoCapture(0)


def load_embeddings(folder):
    embeddings = []
    for file in os.listdir(folder):
        if file.endswith(".npy"):
            emb = np.load(os.path.join(folder, file))
            embeddings.append(emb)

    if len(embeddings) == 0:
        return None, None

    mean_emb = np.mean(embeddings, axis=0)
    return embeddings, mean_emb


folder = "./dataset/embeddings/person"
embeddings, mean_emb = load_embeddings(folder)


def get_face_embedding(frame):
    faces = app.get(frame)

    if len(faces) == 0:
        return None

    # better later: pick largest face
    return faces[0].embedding


def cosine_similarity(a, b):
    return np.dot(a, b) / (norm(a) * norm(b))


def is_same_person(new_emb, embeddings, mean_emb, threshold=0.5):

    best_score = max([cosine_similarity(new_emb, e) for e in embeddings])
    mean_score = cosine_similarity(new_emb, mean_emb)

    # print("Best:", best_score)
    # print("Mean:", mean_score)

    if best_score > 0.5 and mean_score > 0.45:
        return True
    else:
        return False


last_check_time = 0
interval = 0.5  # 500 ms
while True:

    current_time = time.time()

    if current_time - last_check_time >= interval:
        last_check_time = current_time

        ret, frame = cap.read()

        if not ret:
            break

        face_embedding = get_face_embedding(frame)

        if face_embedding is None:
            print("No face")
        else:
            print("Face detected")
            if is_same_person(face_embedding, embeddings, mean_emb):
                print("Same person")
            else:
                print("Different person")

        cv2.imshow("Webcam", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break


cap.release()
cv2.destroyAllWindows()
