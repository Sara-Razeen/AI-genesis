import os
import subprocess

RESUME_PDF_FOLDER = None  # auto detect if None
JOB_DESCRIPTION_FILE = "job.txt"
RESUME_CHUNKS_OUT = "resume_chunks.json"
JD_CHUNKS_OUT = "job_description_chunks.json"
PROCESS_SCRIPT = "preprocess.py"
EMBEDDING_SCRIPT = "embed_resume.py"

def run_command(cmd):
    print(f"\nRunning: {cmd}\n")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print("\n Error running command:", cmd)
        exit(1)
    print("\n Done")

def find_resume_folder():
    folders = [f for f in os.listdir(".") if os.path.isdir(f)]
    for f in folders:
        if any(file.endswith(".pdf") for file in os.listdir(f)):
            print(f" Detected resume folder: {f}")
            return f
    return None


def main():

    global RESUME_PDF_FOLDER

    # ----  Auto-detect resume folder ----
    if RESUME_PDF_FOLDER is None:
        RESUME_PDF_FOLDER = find_resume_folder()

        if not RESUME_PDF_FOLDER:
            print(" No folder with PDF resumes found.")
            print("Place resumes in a folder & run again.")
            exit(1)

    print(" Using resume folder:", RESUME_PDF_FOLDER)

    # ---- Run resume processing ----
    cmd1 = (
        f"python {PROCESS_SCRIPT} {RESUME_PDF_FOLDER} "
        f"--json_out {RESUME_CHUNKS_OUT} "
        f"--job_file {JOB_DESCRIPTION_FILE} "
        f"--job_out {JD_CHUNKS_OUT}"
    )
    run_command(cmd1)

    # ---- Run embeddings ----
    cmd2 = f"python {EMBEDDING_SCRIPT}"
    run_command(cmd2)

    print("\n ALL STEPS COMPLETED!\n")
    print(" Files Created:")
    print(f" - {RESUME_CHUNKS_OUT}")
    print(f" - {JD_CHUNKS_OUT}")
    print(" - resume_embeddings.json")
    print(" - jd_embeddings.json")


if __name__ == "__main__":
    main()










# import os
# import subprocess

# # -------- CONFIG --------
# RESUME_PDF_FOLDER = None                     # Folder containing PDFs
# JOB_DESCRIPTION_FILE = "job.txt"                 # Job description .txt file
# RESUME_CHUNKS_OUT = "resume_chunks.json"
# JD_CHUNKS_OUT = "job_description_chunks.json"
# # Files (your existing scripts)
# PROCESS_SCRIPT = "preprocess.py"
# EMBEDDING_SCRIPT = "embed_resume.py"
# # ------------------------


# def run_command(cmd):
#     print(f"\nRunning: {cmd}\n")
#     result = subprocess.run(cmd, shell=True)
#     if result.returncode != 0:
#         print("\n Error running command:", cmd)
#         exit(1)
#     print("\n Done")


# def main():

#     # 1️ Check resume folder
#     if not os.path.exists(RESUME_PDF_FOLDER):
#         print(f" Resume folder not found: {RESUME_PDF_FOLDER}")
#         exit(1)

#     print(" Resume folder found:", RESUME_PDF_FOLDER)

#     # 2️ Run resume processing & chunking
#     cmd1 = (
#         f"python {PROCESS_SCRIPT} {RESUME_PDF_FOLDER} "
#         f"--json_out {RESUME_CHUNKS_OUT} "
#         f"--job_file {JOB_DESCRIPTION_FILE} "
#         f"--job_out {JD_CHUNKS_OUT}"
#     )
#     run_command(cmd1)

#     # 3️ Run embeddings generation
#     cmd2 = f"python {EMBEDDING_SCRIPT}"
#     run_command(cmd2)

#     print("\n ALL STEPS COMPLETED SUCCESSFULLY!\n")
#     print(" OUTPUT FILES CREATED:")
#     print(f"  - {RESUME_CHUNKS_OUT}")
#     print(f"  - {JD_CHUNKS_OUT}")
#     print(f"  - resume_embeddings.json")
#     print(f"  - jd_embeddings.json")


# if __name__ == "__main__":
#     main()







# from fastapi import FastAPI, File, UploadFile, Form
# from fastapi.responses import JSONResponse
# from preprocess import extract_text_pdf, clean_text, chunk_resume
# from sentence_transformers import SentenceTransformer
# import json
# import tempfile

# app = FastAPI(title="Resume Embedding API")

# # Load model globally (for performance)
# model = SentenceTransformer("BAAI/bge-base-en")

# @app.post("/embed")
# async def embed_resume(
#     resume_pdf: UploadFile = File(...),
#     job_text: str = Form(None)
# ):
#     # Save uploaded PDF temporarily
#     with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
#         tmp.write(await resume_pdf.read())
#         pdf_path = tmp.name

#     # Extract and clean text
#     resume_text = extract_text_pdf(pdf_path)
#     clean_resume = clean_text(resume_text)
#     resume_chunks = chunk_resume(clean_resume)

#     # Embed resume chunks
#     resume_embeddings = [model.encode(c).tolist() for c in resume_chunks]

#     response_data = {
#         "resume_file": resume_pdf.filename,
#         "resume_chunks": resume_chunks,
#         "resume_embeddings": resume_embeddings
#     }

#     # Optional: job description
#     if job_text:
#         clean_job = clean_text(job_text)
#         job_chunks = chunk_resume(clean_job)
#         job_embeddings = [model.encode(c).tolist() for c in job_chunks]

#         response_data["job_chunks"] = job_chunks
#         response_data["job_embeddings"] = job_embeddings

#     return JSONResponse(content=response_data)


# @app.get("/")
# async def root():
#     return {"message": "Resume Embedding API is running. Use POST /embed"}