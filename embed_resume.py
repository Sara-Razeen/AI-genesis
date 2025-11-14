import json
from sentence_transformers import SentenceTransformer
import numpy as np

# ------------ Load embedding model ------------
model = SentenceTransformer("BAAI/bge-base-en")

# ========== 1️ Process Resume Chunks ==========
with open("resume_chunks.json", "r", encoding="utf-8") as f:
    resume_data = json.load(f)

resume_embeddings = []
for item in resume_data:
    vector = model.encode(item["text"]).tolist()

    resume_embeddings.append({
        "resume_id": item["resume_id"],
        "file_name": item["file_name"],
        "chunk_id": item["chunk_id"],
        "text": item["text"],
        "embedding": vector
    })

with open("resume_embeddings.json", "w", encoding="utf-8") as f:
    json.dump(resume_embeddings, f, indent=4)

print(f"Resume embeddings generated for {len(resume_embeddings)} chunks and saved to resume_embeddings.json")
print("Vector Length:", len(vector))

# ========== 2️ Process Job Description Chunks ==========
# with open("job_description_chunks.json", "r", encoding="utf-8") as f:
#     jd_data = json.load(f)

# jd_embeddings = []
# for item in jd_data:
#     vector = model.encode(item["text"]).tolist()

#     jd_embeddings.append({
#         "jd_chunk_id": item["jd_chunk_id"],
#         "text": item["text"],
#         "embedding": vector
#     })

# with open("jd_embeddings.json", "w", encoding="utf-8") as f:
#     json.dump(jd_embeddings, f, indent=4)

# print(f"Job description embeddings generated for {len(jd_embeddings)} chunks and saved to jd_embeddings.json")
# print("Vector Length:", len(vector))



