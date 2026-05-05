import cv2
from insightface.app import FaceAnalysis
import numpy as np
from numpy.linalg import norm
import os
import time
from core.database import load_database
from core.detector import detect_faces
from core.matcher import find_person

app = FaceAnalysis()
app.prepare(ctx_id=0)
cap = cv2.VideoCapture(0)


folder = "./dataset/embeddings/"

database = load_database(folder)


def cosine_similarity(a, b):
    return np.dot(a, b) / (norm(a) * norm(b))


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

        face_embedding = detect_faces(frame)

        if len(face_embedding) == 0:
            last_face_embedding = None
            last_rect = None
            last_person = None
        else:

            rect = face_embedding[0].bbox.astype(int)

            last_face_embedding = face_embedding[0].embedding
            last_rect = rect
            last_person = find_person(face_embedding[0].embedding, database)

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
