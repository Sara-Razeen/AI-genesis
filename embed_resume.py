import sys
sys.stdout.reconfigure(encoding='utf-8')

import os
import json
from dotenv import load_dotenv, find_dotenv
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, PointStruct

# ------------------ Load environment variables ------------------
load_dotenv(find_dotenv())

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "cv_vectors")

print("DEBUG:", QDRANT_URL, COLLECTION_NAME)

# ------------------ Load embedding model ------------------
model = SentenceTransformer("BAAI/bge-base-en")
VECTOR_SIZE = model.get_sentence_embedding_dimension()  # 768

# ------------------ Load resume chunks ------------------
with open("resume_chunks.json", "r", encoding="utf-8") as f:
    resume_data = json.load(f)

print(f"Loaded {len(resume_data)} resume chunks")

# ------------------ Process chunks → Generate embeddings ------------------
resume_embeddings = []
points = []

# for idx, item in enumerate(resume_data):

#     vector = model.encode(item["text"]).tolist()



#     # Save to JSON backup

#     resume_embeddings.append({

#         "resume_id": item["resume_id"],

#         "file_name": item["file_name"],

#         "chunk_id": item["chunk_id"],

#         "text": item["text"],

#         "embedding": vector

#     })



#     # Prepare Qdrant point

#     points.append(

#         PointStruct(

#             id=idx,

#             vector=vector,

#             payload={

#                 "resume_id": item["resume_id"],

#                 "file_name": item["file_name"],

#                 "chunk_id": item["chunk_id"],

#                 "text": item["text"]

#             },

#         )

#     )

for idx, item in enumerate(resume_data):
    
    # FIX 1: Get the text from "page_content" instead of "text"
    page_content = item.get("page_content", "")
    
    # FIX 2: Get the metadata block
    metadata = item.get("metadata", {})

    # Skip if page_content is empty (handles filter just in case)
    if not page_content:
        print(f"Skipping empty chunk at index {idx}")
        continue
        
    vector = model.encode(page_content).tolist()

    # Save to JSON backup (updated to new structure)
    resume_embeddings.append({
        "page_content": page_content,
        "metadata": metadata,
        "embedding": vector
    })

    # Prepare Qdrant point
    # The payload MUST match what LangChain expects:
    # A "page_content" key and a "metadata" key.
    payload = {
        "page_content": page_content,
        "metadata": metadata
    }
    
    points.append(
        PointStruct(
            id=idx,
            vector=vector,
            payload=payload # FIX 3: Use the new payload structure
        )
    )

# Save embeddings locally (optional but good for debugging)
with open("resume_embeddings.json", "w", encoding="utf-8") as f:
    json.dump(resume_embeddings, f, indent=4)

print(f"Saved embeddings to resume_embeddings.json")
print(f"Vector dimension: {VECTOR_SIZE}")

# ------------------ Connect to Qdrant ------------------
client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

# ------------------ Recreate collection safely ------------------
if client.collection_exists(COLLECTION_NAME):
    print(f"Collection '{COLLECTION_NAME}' exists. Deleting...")
    client.delete_collection(COLLECTION_NAME)

print(f"Creating collection '{COLLECTION_NAME}'...")
client.create_collection(
    collection_name=COLLECTION_NAME,
    vectors_config=VectorParams(
        size=VECTOR_SIZE,
        distance="Cosine",
    )
)
# ------------------ Upload vectors to Qdrant ------------------
client.upsert(collection_name=COLLECTION_NAME, points=points)

# ------------------ Count vectors in DB ------------------
count = client.count(COLLECTION_NAME).count
print(f"Uploaded {count} vectors into collection '{COLLECTION_NAME}'")
