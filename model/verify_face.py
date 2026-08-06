import cv2
import numpy as np
from insightface.app import FaceAnalysis


# Load InsightFace Model (Only Once)


print("Loading InsightFace...")

app = FaceAnalysis(name="buffalo_l")

app.prepare(
    ctx_id=-1,
    det_size=(1280, 1280)
)

print("InsightFace Loaded Successfully")



# Face Verification Function


def verify_face(selfie_path, document_path):
    THRESHOLD = 0.60

  
    # Read Images
    

    selfie = cv2.imread(selfie_path)
    document = cv2.imread(document_path)

    if selfie is None:
        raise Exception("Unable to read selfie image.")

    if document is None:
        raise Exception("Unable to read document image.")

  
    # Detect Faces


    selfie_faces = app.get(selfie)
    document_faces = app.get(document)

    if len(selfie_faces) == 0:
        raise Exception("No face detected in selfie.")

    if len(document_faces) == 0:
        raise Exception("No face detected in document.")

  
    # Extract Face Embeddings


    selfie_embedding = selfie_faces[0].embedding
    document_embedding = document_faces[0].embedding

 
    # Calculate Cosine Similarity


    similarity = np.dot(
        selfie_embedding,
        document_embedding
    ) / (
        np.linalg.norm(selfie_embedding)
        * np.linalg.norm(document_embedding)
    )


    # Face Verification


    if similarity >= THRESHOLD:
        face_verified = True
    else:
        face_verified = False

    return float(similarity), face_verified