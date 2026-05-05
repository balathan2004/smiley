import cv2
from insightface.app import FaceAnalysis
import numpy as np
from numpy.linalg import norm
import os
import time

from core.database import load_person_embeddings
from core.detector import detect_faces
from core.matcher import is_certain_match

filepath = "./faces/v1.mp4"

cap = cv2.VideoCapture(filepath)

person = "pooja"


folder = "./dataset/embeddings/" + person
embeddings, mean_emb = load_person_embeddings(folder)


def get_face_embedding(frame):
    faces = detect_faces(frame)

    if len(faces) == 0:
        return None

    # better later: pick largest face
    return faces


def cosine_similarity(a, b):
    return np.dot(a, b) / (norm(a) * norm(b))


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

                if is_certain_match(person, face.embedding):
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
