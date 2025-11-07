import re
from pathlib import Path
import pdfplumber
import fitz  # PyMuPDF
from langchain_text_splitters import RecursiveCharacterTextSplitter
import json

# ------------ PDF Extraction ------------
def extract_text_pdf(path: str) -> str:
    path = Path(path)
    text = ""

    # Try pdfplumber (best for text PDFs)
    try:
        with pdfplumber.open(path) as pdf:
            pages = [p.extract_text() or "" for p in pdf.pages]
            text = "\n\n".join(pages).strip()
    except:
        pass

    # If no text, try PyMuPDF (works on many PDFs)
    if not text or len(text) < 20:
        try:
            doc = fitz.open(path)
            pages = [page.get_text("text") for page in doc]
            doc.close()
            text = "\n\n".join(pages).strip()
        except:
            pass

    return text

# ------------ Cleaning ------------
def clean_text(text: str) -> str:
    if not text:
        return ""

    # text = text.lower()
    text = text.replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"\s+", " ", text)              # remove extra spaces
    # text = re.sub(r"https?://\S+|www\.\S+", "", text)
    # text = re.sub(r"\S+@\S+", "", text)           # remove emails
    # text = re.sub(r"[^\x00-\x7F]+", " ", text)    # remove weird chars
    # text = re.sub(r'[^a-z0-9@.,\-\s]', '', text)
    # text = text.strip()

    return text

# ------------ Process All Resumes & Save in One File ------------
def process_folder_to_single_file(folder_path: str, out_file: str = "cleaned_all.txt"):
    folder = Path(folder_path)
    pdf_files = list(folder.glob("*.pdf"))

    if not pdf_files:
        print(" No PDF resumes found in the folder.")
        return

    with open(out_file, "w", encoding="utf-8") as output:
        for idx, pdf in enumerate(pdf_files):
            resume_id = idx + 1
            print(f"Processing: {pdf.name}")
            raw = extract_text_pdf(pdf)
            cleaned_text = clean_text(raw)

            output.write(f"\n\n")
            output.write(f"RESUME ID: {resume_id}\n")
            output.write(f"FILE NAME: {pdf.name}\n")
            output.write(cleaned_text)
            output.write("\n\n")
            

    print(f"\n All resumes cleaned & stored in: {out_file}")

# Chunking using LangChain
def chunk_resume(cleaned_text, chunk_size=475, chunk_overlap=25):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,#to avoid context loss
        length_function=len
    )
    chunks = splitter.split_text(cleaned_text)
    return chunks

# Save Chunks to JSON file
def save_chunks(chunks, output_path="resume_chunks.json"):
    data = [{"chunk_id": i+1, "text": chunk} for i, chunk in enumerate(chunks)]
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    print(f"Saved {len(chunks)} chunks to {output_path}")

# ------------ CLI ------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Combine cleaned resumes into one file")
    parser.add_argument("folder", help="Folder containing PDF resumes")
    parser.add_argument("--json_out", help="Output chunk JSON file", default="resume_chunks.json")

    parser.add_argument("--job_file", help="Path to job description .txt file", default=None)
    parser.add_argument("--job_out", help="Output chunk JSON for JD", default="job_description_chunks.json")

    args = parser.parse_args()

    folder = Path(args.folder)
    pdf_files = list(folder.glob("*.pdf"))

    # ---------  Auto-detect resume folder ---------
    if args.folder:
        folder = Path(args.folder)
    else:
        print(" Detecting resume folder automatically...")
        folders = [f for f in Path(".").iterdir() if f.is_dir()]

        folder = None
        for f in folders:
            if list(Path(f).glob("*.pdf")):
                folder = Path(f)
                print(f" Found folder with resumes: {f}")
                break

        if folder is None:
            print(" No folder containing PDF resumes found.")
            print("Hint: place resumes in a folder or use --folder <path>")
            exit()

    pdf_files = list(folder.glob("*.pdf"))
    if not pdf_files:
        print(f" No PDF files found in folder: {folder}")
        exit()

    process_folder_to_single_file(args.folder, "cleaned_all.txt")

    all_chunks = []
    
    for idx, pdf in enumerate(pdf_files):
        resume_id = idx + 1
        print(f"Processing: {pdf.name}")

        raw_text = extract_text_pdf(pdf)
        clean = clean_text(raw_text)
        chunks = chunk_resume(clean)

        for i, c in enumerate(chunks):
            all_chunks.append({
                "resume_id": resume_id,
                "file_name": pdf.name,
                "chunk_id": i + 1,
                "text": c
            })

    with open(args.json_out, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, indent=4, ensure_ascii=False)

    print(f"Finished! Saved {len(all_chunks)} chunks to {args.json_out}")

    # --------- Process Job Description (if provided) ---------
    if args.job_file:
        print(f"\nProcessing Job Description File: {args.job_file}")

        with open(args.job_file, "r", encoding="utf-8") as jf:
            jd_text = jf.read()

        jd_clean = clean_text(jd_text)
        jd_chunks = chunk_resume(jd_clean)

        jd_chunk_data = [
            {"jd_chunk_id": i + 1, "text": chunk}
            for i, chunk in enumerate(jd_chunks)
        ]

        with open(args.job_out, "w", encoding="utf-8") as f:
            json.dump(jd_chunk_data, f, indent=4, ensure_ascii=False)

        print(f"Job description chunked into {len(jd_chunks)} parts and saved to {args.job_out}")



