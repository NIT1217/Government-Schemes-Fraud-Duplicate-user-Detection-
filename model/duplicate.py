import cv2
import chromadb
from insightface.app import FaceAnalysis


# Configuration


CHROMA_DB_PATH = r"C:\Users\HP\Desktop\Government Scheme Application\Government-Schemes-Fraud-Duplicate-user-Detection-\Database"

COLLECTION_NAME = "face_embeddings"

SIMILARITY_THRESHOLD = 0.85

TOP_K = 5

# Load ChromaDB (Only Once)


print("Loading ChromaDB...")

client = chromadb.PersistentClient(path=CHROMA_DB_PATH)

collection = client.get_collection(COLLECTION_NAME)

print("ChromaDB Loaded Successfully")

# Load InsightFace (Only Once)


print("Loading InsightFace...")

app = FaceAnalysis(name="buffalo_l")

app.prepare(ctx_id=-1)

print("InsightFace Loaded Successfully")


# Duplicate Detection Function


def check_duplicate(image_path):


    # Read Image
    img = cv2.imread(image_path)

    if img is None:
        raise Exception("Unable to read image.")

    # Detect Face
    faces = app.get(img)

    if len(faces) == 0:
        raise Exception("No face detected.")

    embedding = faces[0].embedding.tolist()

    # Search ChromaDB
    results = collection.query(

        query_embeddings=[embedding],

        n_results=TOP_K

    )

    ids = results["ids"][0]
    distances = results["distances"][0]
    metadatas = results["metadatas"][0]

    duplicate_found = False

    best_similarity = 0

    best_match = None

    print("Top Matches")


    for rank in range(len(ids)):

        similarity = 1 - distances[rank]

        metadata = metadatas[rank]

        print(f"\nRank : {rank + 1}")
        print("Person :", metadata["person"])
        print("Dataset :", metadata["dataset"])
        print("Image :", metadata["image"])
        print("Similarity :", round(similarity, 4))

        if similarity > best_similarity:

            best_similarity = similarity

            best_match = metadata

        if similarity >= SIMILARITY_THRESHOLD:

            duplicate_found = True

    return duplicate_found, best_similarity, best_match


