import json
from sentence_transformers import SentenceTransformer
import numpy as np

# ------------ Load embedding model ------------
model = SentenceTransformer("BAAI/bge-base-en")

# ========== 1️ Process Resume Chunks ==========
# with open("resume_chunks.json", "r", encoding="utf-8") as f:
#     resume_data = json.load(f)

# resume_embeddings = []
# for item in resume_data:
#     vector = model.encode(item["text"]).tolist()

#     resume_embeddings.append({
#         "resume_id": item["resume_id"],
#         "file_name": item["file_name"],
#         "chunk_id": item["chunk_id"],
#         "text": item["text"],
#         "embedding": vector
#     })

# with open("resume_embeddings.json", "w", encoding="utf-8") as f:
#     json.dump(resume_embeddings, f, indent=4)

# print(f"Resume embeddings generated for {len(resume_embeddings)} chunks and saved to resume_embeddings.json")
# print("Vector Length:", len(vector))


from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

import os
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION")
print("DEBUG:", QDRANT_URL, COLLECTION_NAME)

import json
import os
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, PointStruct
from dotenv import load_dotenv

# ------------------ Load environment ------------------
load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "cv_vectors")

# ------------------ Load embedding model ------------------
model = SentenceTransformer("BAAI/bge-base-en")
VECTOR_SIZE = model.get_sentence_embedding_dimension()   # should be 768

# ------------------ Load chunks ------------------
with open("resume_chunks.json", "r", encoding="utf-8") as f:
    resume_data = json.load(f)

print(f"Loaded {len(resume_data)} resume chunks")

# ------------------ Connect to Qdrant ------------------
client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

# ------------------ Recreate collection ------------------
print(f"Recreating collection '{COLLECTION_NAME}' with vector size {VECTOR_SIZE}...")

client.recreate_collection(
    collection_name=COLLECTION_NAME,
    vectors_config=VectorParams(
        size=VECTOR_SIZE,
        distance="Cosine",
    )
)

# ------------------ Generate embeddings + upload ------------------
points = []
for idx, item in enumerate(resume_data):
    vector = model.encode(item["text"]).tolist()

    points.append(
        PointStruct(
            id=idx,
            vector=vector,
            payload={
                "resume_id": item["resume_id"],
                "file_name": item["file_name"],
                "chunk_id": item["chunk_id"],
                "text": item["text"]
            },
        )
    )

# Upload to Qdrant
client.upsert(collection_name=COLLECTION_NAME, points=points)

# ------------------ Count vectors ------------------
count = client.count(COLLECTION_NAME).count
print(f"Uploaded {count} vectors into Qdrant collection: {COLLECTION_NAME}")
print(f"Vector size used: {VECTOR_SIZE}")
