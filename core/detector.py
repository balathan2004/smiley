from insightface.app import FaceAnalysis

app = FaceAnalysis()
app.prepare(ctx_id=0)


def detect_faces(frame):
    return app.get(frame)
