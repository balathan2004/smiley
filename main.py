import os
import cv2
import numpy as np
from insightface.app import FaceAnalysis


person = "violet"

# 🔧 Config
INPUT_FOLDER = "./faces/" + person
EMB_FOLDER = "./dataset/embeddings/" + person
# FACE_FOLDER = "./dataset/faces/" + person

os.makedirs(EMB_FOLDER, exist_ok=True)
# os.makedirs(FACE_FOLDER, exist_ok=True)


# 🚀 Load model
app = FaceAnalysis(providers=['CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))


for filename in os.listdir(INPUT_FOLDER):
    if not filename.lower().endswith((".png", ".jpg", ".jpeg")):
        continue

    img_path = os.path.join(INPUT_FOLDER, filename)
    img = cv2.imread(img_path)

    if img is None:
        print(f"Skipping {filename} ❌")
        continue    

    faces = app.get(img)

    if len(faces) == 0: 
        print(f"No face in {filename} ⚠️")
        continue

    # 👉 Take first face (assumes 1 person per image)
    face = faces[0]

    # 🔐 Save embedding
    embedding = face.embedding
    name = os.path.splitext(filename)[0]

    np.save(os.path.join(EMB_FOLDER, f"{name}.npy"), embedding)

    # ✂️ Save cropped face (optional)
    x1, y1, x2, y2 = face.bbox.astype(int)

    h, w, _ = img.shape
    pad = 20
    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(w, x2 + pad)
    y2 = min(h, y2 + pad)

    # face_crop = img[y1:y2, x1:x2]
    # cv2.imwrite(os.path.join(FACE_FOLDER, f"{name}.jpg"), face_crop)

    print(f"Processed {filename} ✅")

print("Dataset creation done 🎉")