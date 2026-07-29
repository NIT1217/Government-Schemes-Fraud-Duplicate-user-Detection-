import cv2
import numpy as np
from insightface.app import FaceAnalysis

# -----------------------------
# Load InsightFace Model (Only Once)
# -----------------------------

print("Loading InsightFace...")

app = FaceAnalysis(name="buffalo_l")

app.prepare(
    ctx_id=-1,
    det_size=(1280, 1280)
)

print("InsightFace Loaded Successfully")


# -----------------------------
# Face Verification Function
# -----------------------------

def verify_face(selfie_path, document_path):
    """
    Verifies whether the selfie matches
    the face present on the identity document.

    Returns:
        similarity (float)
        face_verified (bool)
    """

    THRESHOLD = 0.60

    # -----------------------------
    # Read Images
    # -----------------------------

    selfie = cv2.imread(selfie_path)
    document = cv2.imread(document_path)

    if selfie is None:
        raise Exception("Unable to read selfie image.")

    if document is None:
        raise Exception("Unable to read document image.")

    # -----------------------------
    # Detect Faces
    # -----------------------------

    selfie_faces = app.get(selfie)
    document_faces = app.get(document)

    if len(selfie_faces) == 0:
        raise Exception("No face detected in selfie.")

    if len(document_faces) == 0:
        raise Exception("No face detected in document.")

    # -----------------------------
    # Extract Face Embeddings
    # -----------------------------

    selfie_embedding = selfie_faces[0].embedding
    document_embedding = document_faces[0].embedding

    # -----------------------------
    # Calculate Cosine Similarity
    # -----------------------------

    similarity = np.dot(
        selfie_embedding,
        document_embedding
    ) / (
        np.linalg.norm(selfie_embedding)
        * np.linalg.norm(document_embedding)
    )

    # -----------------------------
    # Face Verification
    # -----------------------------

    if similarity >= THRESHOLD:
        face_verified = True
    else:
        face_verified = False

    return float(similarity), face_verified


# -----------------------------
# Testing
# -----------------------------

if __name__ == "__main__":

    SELFIE_IMAGE = r"C:\Users\HP\Desktop\Government Scheme Application\Government-Schemes-Fraud-Duplicate-user-Detection-\testdata\Balvinder PAN.jpg"

    DOCUMENT_IMAGE = r"C:\Users\HP\Desktop\Government Scheme Application\Government-Schemes-Fraud-Duplicate-user-Detection-\testdata\Balvinder PAN.jpg"

    similarity, verified = verify_face(
        SELFIE_IMAGE,
        DOCUMENT_IMAGE
    )

    print("\n==============================")
    print("Verification Report")
    print("==============================")

    print(f"Similarity Score : {similarity:.4f}")
    print(f"Threshold        : 0.60")
    print(f"Face Verified    : {verified}")

    if verified:
        print("Status           : VERIFIED")
    else:
        print("Status           : FAILED")