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









