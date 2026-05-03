import cv2
from insightface.app import FaceAnalysis
import numpy as np
from numpy.linalg import norm
import os
import time

filepath = "./faces/v1.mp4"


app = FaceAnalysis()
app.prepare(ctx_id=0)
cap = cv2.VideoCapture(filepath)


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


folder = "./dataset/embeddings/violet"
embeddings, mean_emb = load_embeddings(folder)


def get_face_embedding(frame):
    faces = app.get(frame)

    if len(faces) == 0:
        return None

    # better later: pick largest face
    return faces


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


fourcc = cv2.VideoWriter_fourcc(*"mp4v")
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
out = cv2.VideoWriter("output.avi", fourcc, fps, (width, height))


last_check_time = 0
interval = 0.1  # 500 ms

while True:

    ret, frame = cap.read()
    if not ret:
        break

    current_time = time.time()

    frame = cv2.resize(frame, (width, height))

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    if current_time - last_check_time >= interval:
        last_check_time = current_time

        face_embedding = get_face_embedding(frame)

        if face_embedding is None:
            print("No face")

        else:
            print("Face detected")

            for face in face_embedding:
                bbox = face.bbox.astype(int)
                x1, y1, x2, y2 = bbox

                if is_same_person(face.embedding, embeddings, mean_emb):
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    print("Same person")
                else:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    print("Different person")

    out.write(frame)

    # cv2.imshow("Video", frame)

out.release()
cap.release()
cv2.destroyAllWindows()
