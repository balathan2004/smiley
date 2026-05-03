import cv2
from insightface.app import FaceAnalysis
import numpy as np
from numpy.linalg import norm
import os
import time

app = FaceAnalysis()
app.prepare(ctx_id=0)
cap = cv2.VideoCapture(0)


def load_all_embeddings(base_folder):
    database = {}

    for person in os.listdir(base_folder):
        person_folder = os.path.join(base_folder, person)

        if not os.path.isdir(person_folder):
            continue

        embeddings = []

        for file in os.listdir(person_folder):
            if file.endswith(".npy"):
                emb = np.load(os.path.join(person_folder, file))
                embeddings.append(emb)

        if len(embeddings) > 0:
            database[person] = {
                "embeddings": embeddings,
                "mean_emb": np.mean(embeddings, axis=0),
            }

    return database


folder = "./dataset/embeddings/"

database = load_all_embeddings(folder)


def get_face_embedding(frame):
    faces = app.get(frame)
    rect = None

    if len(faces) == 0:
        return None, rect

    # better later: pick largest face
    rect = faces[0].bbox.astype(int)
    return faces[0].embedding, rect


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


def find_person(new_emb, database, threshold=0.5):
    best_person = None
    best_score = 0

    for person, data in database.items():
        embeddings = data["embeddings"]
        mean_emb = data["mean_emb"]

        score = max([cosine_similarity(new_emb, e) for e in embeddings])
        mean_score = cosine_similarity(new_emb, mean_emb)

        if score > best_score and score > threshold and mean_score > threshold:
            best_score = score
            best_person = person

    return best_person


last_face_embedding = None
last_rect = None
last_person = None

last_check_time = 0
interval = 1  # 500 ms
while True:

    ret, frame = cap.read()

    if not ret:
        break

    current_time = time.time()

    if current_time - last_check_time >= interval:
        last_check_time = current_time

        face_embedding, rect = get_face_embedding(frame)

        if face_embedding is None:
            last_face_embedding = None
            last_rect = None
            last_person = None
        else:

            last_face_embedding = face_embedding
            last_rect = rect
            last_person = find_person(face_embedding, database)

    if last_rect is not None:
        x1, y1, x2, y2 = last_rect

    if last_person is not None:
        label = last_person
        color = (0, 255, 0)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        cv2.putText(
            frame,
            label,
            (x1, max(y1 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            color,
            2,
        )

    else:
        label = "Unknown"
        color = (0, 0, 255)

    cv2.imshow("Webcam", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()
