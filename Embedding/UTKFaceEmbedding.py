import os
import uuid
import cv2
import numpy as np
import chromadb
from tqdm import tqdm
from insightface.app import FaceAnalysis

# ============================================================
# CONFIGURATION
# ============================================================

DATASET_PATH = r"C:\Users\HP\Desktop\Government Scheme Application\Government-Schemes-Fraud-Duplicate-user-Detection-\Datasets\UTKFace"

CHROMA_DB_PATH = r"./Database/FaceDB"

COLLECTION_NAME = "face_embeddings"

# ============================================================
# LOAD CHROMADB
# ============================================================

print("=" * 60)
print("Loading ChromaDB...")

client = chromadb.PersistentClient(path=CHROMA_DB_PATH)

collection = client.get_or_create_collection(
    name=COLLECTION_NAME
)

print("ChromaDB Loaded Successfully")

# ============================================================
# LOAD INSIGHTFACE
# ============================================================

print("\nLoading InsightFace Model...")

app = FaceAnalysis(
    name="buffalo_l",
    providers=["CPUExecutionProvider"]
)

app.prepare(
    ctx_id=-1,
    det_size=(640, 640)
)

print("InsightFace Loaded Successfully")

# ============================================================
# COUNTERS
# ============================================================

total_images = 0
successful = 0
failed = 0

# ============================================================
# VALID IMAGE EXTENSIONS
# ============================================================

VALID_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")

# ============================================================
# PROCESS DATASET
# ============================================================

print("\nStarting Embedding Extraction...\n")

image_files = [
    f for f in os.listdir(DATASET_PATH)
    if f.lower().endswith(VALID_EXTENSIONS)
]

for image_name in tqdm(image_files):

    total_images += 1

    image_path = os.path.join(DATASET_PATH, image_name)

    try:

        image = cv2.imread(image_path)

        if image is None:
            failed += 1
            continue

        faces = app.get(image)

        if len(faces) == 0:
            failed += 1
            continue

        # Largest detected face
        face = max(
            faces,
            key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1])
        )

        embedding = face.embedding.astype(np.float32)
        embedding = embedding / np.linalg.norm(embedding)

        unique_id = str(uuid.uuid4())

        collection.add(
            ids=[unique_id],
            embeddings=[embedding.tolist()],
            metadatas=[{
                "dataset": "UTKFace",
                "person_name": os.path.splitext(image_name)[0],
                "image_name": image_name,
                "image_path": image_path
            }]
        )

        successful += 1

    except Exception as e:

        failed += 1
        print(f"Error processing {image_name}: {e}")

# ============================================================
# FINAL REPORT
# ============================================================

print("\n" + "=" * 60)
print("Embedding Extraction Complete")
print("=" * 60)

print(f"Total Images            : {total_images}")
print(f"Successful              : {successful}")
print(f"Failed                  : {failed}")
print(f"Total Embeddings Stored : {collection.count()}")

print("=" * 60)