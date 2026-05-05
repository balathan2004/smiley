import numpy as np
from numpy.linalg import norm
import os
import cv2
from insightface.app import FaceAnalysis

from core.detector import detect_faces
from core.matcher import is_certain_match

person = "pooja"


# 📸 Load new image
test_img = cv2.imread("./faces/test/pooja.jpg")

if test_img is None:
    print("Test image not found ❌")
    exit()

faces = detect_faces(test_img)

if len(faces) == 0:
    print("No face found ❌")
    exit()

# 👉 This is your missing piece
new_emb = faces[0].embedding


is_certain = is_certain_match(person, new_emb)

if is_certain:
    print("Same person ✅")
else:
    print("Different person ❌")
