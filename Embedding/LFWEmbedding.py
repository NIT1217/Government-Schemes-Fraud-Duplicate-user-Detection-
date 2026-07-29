import os
import cv2
import chromadb
import numpy as np
from tqdm import tqdm
from insightface.app import FaceAnalysis

# -----------------------------
# PATHS
# -----------------------------
LFW_PATH = r"C:\Users\HP\Desktop\Government Scheme Application\Government-Schemes-Fraud-Duplicate-user-Detection-\Datasets\LFW (Labeled Faces in the Wild)\lfw-deepfunneled"

CHROMA_DB_PATH = r"./Database/FaceDB"
COLLECTION_NAME = "face_embeddings"

# -----------------------------
# LOAD CHROMADB
# -----------------------------
client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
collection = client.get_or_create_collection(name=COLLECTION_NAME)

# -----------------------------
# LOAD INSIGHTFACE MODEL
# -----------------------------
app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
app.prepare(ctx_id=-1, det_size=(640, 640))

total_images = 0
successful = 0
failed = 0

# -----------------------------
# CREATE EMBEDDINGS
# -----------------------------
for person_name in tqdm(os.listdir(LFW_PATH), desc="Processing LFW"):

    person_folder = os.path.join(LFW_PATH, person_name)

    if not os.path.isdir(person_folder):
        continue

    for image_name in os.listdir(person_folder):

        image_path = os.path.join(person_folder, image_name)

        total_images += 1

        try:
            image = cv2.imread(image_path)

            if image is None:
                failed += 1
                continue

            faces = app.get(image)

            if len(faces) == 0:
                failed += 1
                continue

            # Take largest face
            face = max(
                faces,
                key=lambda f: (f.bbox[2]-f.bbox[0]) * (f.bbox[3]-f.bbox[1])
            )

            embedding = face.embedding.astype(np.float32)
            embedding = embedding / np.linalg.norm(embedding)

            collection.add(
                ids=[f"{person_name}_{image_name}"],
                embeddings=[embedding.tolist()],
                metadatas=[{
                    "person_name": person_name,
                    "image_name": image_name,
                    "image_path": image_path
                }]
            )

            successful += 1

        except Exception as e:
            failed += 1
            print(f"Error: {image_path} -> {e}")

# -----------------------------
# FINAL STATS
# -----------------------------
print("\nEmbedding Extraction Complete")
print("=" * 40)
print(f"Total Images : {total_images}")
print(f"Successful   : {successful}")
print(f"Failed       : {failed}")
print(f"Stored in DB : {collection.count()}")