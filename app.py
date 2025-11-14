# from typing import List
# import os
# import subprocess  # Added from model.py
# import shutil  # Added for managing temp folders
# import streamlit as st
# from dotenv import load_dotenv, find_dotenv
# from typing import List
# from PyPDF2 import PdfReader

# # ---------------- LangChain Imports ----------------
# from langchain_community.embeddings import HuggingFaceEmbeddings
# # from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.schema import Document
# from langchain.memory import ConversationBufferMemory
# from langchain.chains import ConversationalRetrievalChain
# from langchain.prompts import PromptTemplate
# from langchain_google_genai import ChatGoogleGenerativeAI

# # ---------------- Qdrant ----------------
# from langchain_community.vectorstores import Qdrant
# from qdrant_client import QdrantClient

# # ---------------- Custom Templates ----------------
# from htmltemplate import css, bot_template, user_template, header_html, stats_html, footer_html

# load_dotenv(find_dotenv())  # load .env file so GOOGLE_API_KEY is available

# # ============= CONSTANTS FROM model.py =============
# # We create a temp folder for uploaded resumes
# RESUME_PDF_FOLDER = "temp_resumes" 
# JOB_DESCRIPTION_FILE = "job.txt"  # We'll save the uploaded JD as this file
# RESUME_CHUNKS_OUT = "resume_chunks.json"
# JD_CHUNKS_OUT = "job_description_chunks.json"
# PROCESS_SCRIPT = "preprocess.py"
# EMBEDDING_SCRIPT = "embed_resume.py"

# # ============= FUNCTIONS FROM model.py =============
# def run_command(cmd):
#     """Runs a shell command and waits for it to complete."""
#     st.info(f"Running: {cmd}")
#     try:
#         # Using subprocess.run with capture_output and text=True
#         result = subprocess.run(
#             cmd, 
#             shell=True, 
#             check=True, 
#             capture_output=True, 
#             text=True
#         )
#         st.text(result.stdout)
#         if result.stderr:
#             st.text(result.stderr)
#         st.success("Command finished.")
#     except subprocess.CalledProcessError as e:
#         # If the command fails, show the error in Streamlit
#         st.error(f"Error running command: {cmd}")
#         st.error(f"Return Code: {e.returncode}")
#         st.error(f"STDOUT: {e.stdout}")
#         st.error(f"STDERR: {e.stderr}")
#         # Stop the app execution if a script fails
#         st.stop()

# # ============= FILE SAVING (New Helper Function) =============
# def save_uploaded_files(pdf_docs, job_desc_file):
#     """Saves uploaded Streamlit files to disk for processing."""
    
#     # --- 1. Save Job Description ---
#     # We save the job.txt file
#     if job_desc_file:
#         with open(JOB_DESCRIPTION_FILE, "wb") as f:
#             f.write(job_desc_file.getbuffer())
    
#     # --- 2. Save Resumes ---
#     # Clean up old temp resume folder if it exists
#     if os.path.exists(RESUME_PDF_FOLDER):
#         shutil.rmtree(RESUME_PDF_FOLDER)
    
#     # Create new temp folder
#     os.makedirs(RESUME_PDF_FOLDER)
    
#     # Save all uploaded PDF files into it
#     for pdf in pdf_docs:
#         file_path = os.path.join(RESUME_PDF_FOLDER, pdf.name)
#         with open(file_path, "wb") as f:
#             f.write(pdf.getbuffer())
    
#     st.info(f"Saved {len(pdf_docs)} resumes to '{RESUME_PDF_FOLDER}'")
#     # st.info(f"Saved job description to '{JOB_DESCRIPTION_FILE}'")

#     # ============= QDRANT VECTOR STORE (Modified) =============
# def get_qdrant_vectorstore():
#     """
#     Connects to an *existing* Qdrant collection that was populated
#     by the embedding.py script.
#     """
#     embeddings = HuggingFaceEmbeddings(
#         model_name="sentence-transformers/all-mpnet-base-v2"
#     )

#     qdrant_url = os.getenv("QDRANT_URL", "https://60545997-4f65-414f-879d-4c1ee500d0c2.europe-west3-0.gcp.cloud.qdrant.io")
#     qdrant_api_key = os.getenv("QDRANT_API_KEY", None)
#     collection_name = os.getenv("QDRANT_COLLECTION", "pdf_chat_collection")

#     client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)

#     # This connects to an existing collection
#     vectorstore = Qdrant(
#         client=client, 
#         collection_name=collection_name, 
#         embeddings=embeddings
#     )
    
#     st.success("Connected to Qdrant vector store!")
#     return vectorstore
# #=============== PDF PROCESSING ===============

# # def extract_text_from_pdfs(pdf_docs) -> str:
# #     """Extract raw text from uploaded PDF files."""
# #     text = ""
# #     for pdf in pdf_docs:
# #         pdf_reader = PdfReader(pdf)
# #         for page in pdf_reader.pages:
# #             if page.extract_text():
# #                 text += page.extract_text()
# #     return text


# # def split_text_into_chunks(text: str) -> List[Document]:
# #     """Split extracted text into overlapping chunks."""
# #     splitter = RecursiveCharacterTextSplitter(
# #         chunk_size=800,
# #         chunk_overlap=100,
# #         separators=["\n", ".", "?", "!"]
# #     )
# #     chunks = splitter.split_text(text)
# #     return [Document(page_content=chunk) for chunk in chunks]



# #=============== QDRANT VECTOR STORE ===============
# # def create_qdrant_vectorstore(text_chunks):
# #     embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

# #     # Qdrant client setup
# #     qdrant_url = os.getenv("QDRANT_URL", "https://60545997-4f65-414f-879d-4c1ee500d0c2.europe-west3-0.gcp.cloud.qdrant.io")
# #     qdrant_api_key = os.getenv("QDRANT_API_KEY", None)
# #     collection_name = os.getenv("QDRANT_COLLECTION", "pdf_chat_collection")

# #     client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)

# #     vectorstore = Qdrant.from_documents(
# #         documents=text_chunks,
# #         embedding=embeddings,
# #         url=qdrant_url,
# #         api_key=qdrant_api_key,
# #         collection_name=collection_name
# #     )

# #     return vectorstore



# #=============== CUSTOM PROMPT TEMPLATE (Ensures complete answers) ===============
# qa_template = """
# You are an expert assistant providing detailed, complete, and accurate answers.

# Question: {question}
# Context: {context}

# Instructions:
# - If the question asks for a specific number of features, points, or reasons, always list **exactly that number**.
# - Include all relevant details from the given context.
# - Use clear numbering or bullet points (e.g., 1., 2., 3.).
# - Do not summarize or skip any point.

# Answer:
# """

# qa_prompt = PromptTemplate(
#     input_variables=["context", "question"],
#     template=qa_template
# )


# #=============== GEMINI LLM + CONVERSATION CHAIN ===============

# def create_conversation_chain(vectorstore):
#     llm = ChatGoogleGenerativeAI(
#         model="gemini-2.0-flash-exp",
#         google_api_key=os.getenv("GOOGLE_API_KEY"),
#         temperature=0.2,
#         max_output_tokens=2048  # Increased for full responses
#     )

#     memory = ConversationBufferMemory(
#         memory_key="chat_history",
#         return_messages=True
#     )

#     chain = ConversationalRetrievalChain.from_llm(
#         llm=llm,
#         retriever=vectorstore.as_retriever(
#             search_type="similarity",
#             search_kwargs={"k": 5}
#         ),
#         memory=memory,
#         combine_docs_chain_kwargs={"prompt": qa_prompt},
#         verbose=True
#     )

#     return chain



# #=============== CHAT HANDLER ===============
# def handle_user_query(user_question):
#     if st.session_state.conversation is None:
#         st.error("Please process documents first.")
#         return

#     try:
#         response = st.session_state.conversation({"question": user_question})
#         st.session_state.chat_history = response["chat_history"]

#         # Display the full chat history
#         for i, msg in enumerate(st.session_state.chat_history):
#             template = user_template if i % 2 == 0 else bot_template
#             st.write(template.replace("{{MSG}}", msg.content), unsafe_allow_html=True)

#     except Exception as e:
#         st.error(f"Error: {str(e)}")


# def clear_chat():
#     st.session_state.chat_history = []
#     if st.session_state.conversation:
#         st.session_state.conversation.memory.clear()
#     st.rerun()


# #=============== STREAMLIT MAIN APP ===============
# def main():
    
#     st.set_page_config(page_title="Chat with PDFs (Gemini + Qdrant)", page_icon="📚", layout="wide")
#     st.markdown(header_html, unsafe_allow_html=True)
#     st.markdown(stats_html, unsafe_allow_html=True)

#     st.write(css, unsafe_allow_html=True)

#     # Initialize session states
#     if "conversation" not in st.session_state:
#         st.session_state.conversation = None
#     if "chat_history" not in st.session_state:
#         st.session_state.chat_history = []

#     # st.header("💬 Resume Screening")
#         # --- Header Section ---
#     st.markdown("""
#         <div style="text-align:center; padding: 1rem 0;">
#             <h1 style="color:#3b82f6; font-family:'Poppins',sans-serif;">💬 Resume Screening Assistant</h1>
#             <p style="color:#94a3b8; font-size:0.95rem;">Chat with your uploaded resumes or PDFs, powered by Gemini + Qdrant</p>
#         </div>
#     """, unsafe_allow_html=True)

#     # Clear chat
#     # col1, col2 = st.columns([3, 1])
#     # with col2:
#     #     if st.button("🧹 Clear Chat"):
#     #         clear_chat()

#        # --- Chat Display Window ---
#     st.markdown("<div class='chat-window'>", unsafe_allow_html=True)
#     if st.session_state.chat_history:
#         for i, msg in enumerate(st.session_state.chat_history):
#             template = user_template if i % 2 == 0 else bot_template
#             st.write(template.replace("{{MSG}}", msg.content), unsafe_allow_html=True)
#     st.markdown("</div>", unsafe_allow_html=True)


#     # # User query
#     # user_question = st.text_input("Ask something about your uploaded documents:")
#     # if user_question:
#     #     handle_user_query(user_question)

#     def main():
#         st.set_page_config(page_title="Chat with PDFs (Gemini + Qdrant)", page_icon="💬", layout="wide")
#         st.write(css, unsafe_allow_html=True)
#         st.markdown(header_html, unsafe_allow_html=True)
#         st.markdown(stats_html, unsafe_allow_html=True)


#         # --- Initialize session state ---
#         if "conversation" not in st.session_state:
#             st.session_state.conversation = None
#         if "chat_history" not in st.session_state:
#             st.session_state.chat_history = []

#         # --- Header Section ---
#         st.markdown("""
#             <div style="text-align:center; padding: 1rem 0;">
#                 <h1 style="color:#3b82f6; font-family:'Poppins',sans-serif;">💬 Resume Screening Assistant</h1>
#                 <p style="color:#94a3b8; font-size:0.95rem;">Chat with your uploaded resumes or PDFs, powered by Gemini + Qdrant</p>
#             </div>
#         """, unsafe_allow_html=True)

#         # --- Chat Display Window ---
#         st.markdown("<div class='chat-window'>", unsafe_allow_html=True)
#         if st.session_state.chat_history:
#             for i, msg in enumerate(st.session_state.chat_history):
#                 template = user_template if i % 2 == 0 else bot_template
#                 st.write(template.replace("{{MSG}}", msg.content), unsafe_allow_html=True)
#         st.markdown("</div>", unsafe_allow_html=True)

#         # --- Input Field & Send Button ---
#         st.markdown("""
#             <style>
#                 .stTextInput>div>div>input {
#                     border-radius: 10px;
#                     background: rgba(30,41,59,0.7);
#                     border: 1px solid rgba(59,130,246,0.4);
#                     color: #e2e8f0;
#                     padding: 0.75rem;
#                     font-size: 0.9rem;
#                 }
#                 .stTextInput>div>div>input:focus {
#                     border-color: #60a5fa;
#                     box-shadow: 0 0 10px rgba(96,165,250,0.4);
#                 }
#                 .stButton>button {
#                     width: 100%;
#                     background: linear-gradient(90deg, #2563eb, #1d4ed8);
#                     color: white;
#                     border: none;
#                     border-radius: 10px;
#                     padding: 0.6rem 1rem;
#                     font-size: 0.95rem;
#                     transition: 0.3s;
#                 }
#                 .stButton>button:hover {
#                     transform: scale(1.03);
#                     background: linear-gradient(90deg, #1d4ed8, #2563eb);
#                 }
#             </style>
#         """, unsafe_allow_html=True)

#     col1, col2 = st.columns([5, 1])
#     with col1:
#         user_question = st.text_input("Type your question here...")
#     with col2:
#         if st.button("🚀 Send"):
#             if user_question:
#                 handle_user_query(user_question)

#         # --- Clear Chat Button ---
#     if st.button("🧹 Clear Chat", key="clear_chat"):
#        clear_chat()

# # --- Sidebar: PDF Upload & Processing ---
# # with st.sidebar:
# #     st.subheader("📂 Upload PDFs")
# #     pdf_docs = st.file_uploader("Upload PDF files here", accept_multiple_files=True, type="pdf")
#     with st.sidebar:
#        st.subheader("📂 Upload Documents")
#        pdf_docs = st.file_uploader(
#         "Upload PDF resumes here", 
#         accept_multiple_files=True, 
#         type="pdf"
#     )

#        if st.button("⚙️ Process Documents"):
#     #  if not pdf_docs or not job_desc_file:
#            if not pdf_docs or not job_desc_file:
#                 st.warning("Please upload at least one PDF resume AND a job description.")
#                 return

#            with st.spinner("Processing and indexing documents..."):
#                 try:
#                     # 1. Save uploaded files to disk
#                     save_uploaded_files(pdf_docs, job_desc_file)
#                     # 2. Run resume processing (from model.py)
#                     cmd1 = (
#                         f"python {PROCESS_SCRIPT} {RESUME_PDF_FOLDER} "
#                         f"--json_out {RESUME_CHUNKS_OUT} "
#                         f"--job_file {JOB_DESCRIPTION_FILE} "
#                         f"--job_out {JD_CHUNKS_OUT}"
#                     )
#                     run_command(cmd1)

#                     # 3. Run embeddings (from model.py)
#                     # This script (embed_resume.py) is expected to
#                     # load the JSON files and populate Qdrant.
#                     cmd2 = f"python {EMBEDDING_SCRIPT}"
#                     run_command(cmd2)

#                     # 4. Connect to the now-populated Qdrant store
#                     vectorstore = get_qdrant_vectorstore()
                    
#                     # 5. Create the conversation chain
#                     st.session_state.conversation = create_conversation_chain(vectorstore)
                    
#                     st.success("✅ Documents processed and loaded successfully. You can now chat!")

#                 except Exception as e:
#                     st.error(f"Error during processing: {str(e)}")


#     st.markdown(footer_html,unsafe_allow_html=True)
# if __name__ == "__main__":
#     main()

from typing import List
import os
import subprocess  # Added from model.py
import shutil  # Added for managing temp folders
import streamlit as st
from dotenv import load_dotenv, find_dotenv
from typing import List
from PyPDF2 import PdfReader

# ---------------- LangChain Imports ----------------
from langchain_community.embeddings import HuggingFaceEmbeddings
# from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

# ---------------- Qdrant ----------------
from langchain_community.vectorstores import Qdrant
from qdrant_client import QdrantClient

# ---------------- Custom Templates ----------------
from htmltemplate import css, bot_template, user_template, header_html, stats_html, footer_html

load_dotenv(find_dotenv())  # load .env file so GOOGLE_API_KEY is available

# ============= CONSTANTS FROM model.py =============
RESUME_PDF_FOLDER = "temp_resumes" 
# JOB_DESCRIPTION_FILE = "job.txt"  # COMMENTED OUT
RESUME_CHUNKS_OUT = "resume_chunks.json"
# JD_CHUNKS_OUT = "job_description_chunks.json" # COMMENTED OUT
PROCESS_SCRIPT = "preprocess.py"
EMBEDDING_SCRIPT = "embed_resume.py"

# ============= FUNCTIONS FROM model.py =============
def run_command(cmd):
    """Runs a shell command and waits for it to complete."""
    st.info(f"Running: {cmd}")
    try:
        result = subprocess.run(
            cmd, 
            shell=True, 
            check=True, 
            capture_output=True, 
            text=True
        )
        st.text(result.stdout)
        if result.stderr:
            st.text(result.stderr)
        st.success("Command finished.")
    except subprocess.CalledProcessError as e:
        st.error(f"Error running command: {cmd}")
        st.error(f"Return Code: {e.returncode}")
        st.error(f"STDOUT: {e.stdout}")
        st.error(f"STDERR: {e.stderr}")
        st.stop()

# ============= FILE SAVING (New Helper Function) =============
def save_uploaded_files(pdf_docs): # Removed job_desc_file
    """Saves uploaded Streamlit files to disk for processing."""
    
    # --- 1. Save Job Description (COMMENTED OUT) ---
    # if job_desc_file:
    #     with open(JOB_DESCRIPTION_FILE, "wb") as f:
    #         f.write(job_desc_file.getbuffer())
    #     st.info(f"Saved job description to '{JOB_DESCRIPTION_FILE}'")
    
    # --- 2. Save Resumes ---
    if os.path.exists(RESUME_PDF_FOLDER):
        shutil.rmtree(RESUME_PDF_FOLDER)
    os.makedirs(RESUME_PDF_FOLDER)
    
    for pdf in pdf_docs:
        file_path = os.path.join(RESUME_PDF_FOLDER, pdf.name)
        with open(file_path, "wb") as f:
            f.write(pdf.getbuffer())
    
    st.info(f"Saved {len(pdf_docs)} resumes to '{RESUME_PDF_FOLDER}'")

# ============= QDRANT VECTOR STORE (Modified) =============
def get_qdrant_vectorstore():
    """
    Connects to an *existing* Qdrant collection that was populated
    by the embedding.py script.
    """
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2"
    )

    qdrant_url = os.getenv("QDRANT_URL", "https://60545997-4f65-414f-879d-4c1ee500d0c2.europe-west3-0.gcp.cloud.qdrant.io")
    qdrant_api_key = os.getenv("QDRANT_API_KEY", None)
    collection_name = os.getenv("QDRANT_COLLECTION", "pdf_chat_collection")

    client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)

    vectorstore = Qdrant(
        client=client, 
        collection_name=collection_name, 
        embeddings=embeddings
    )
    
    st.success("Connected to Qdrant vector store!")
    return vectorstore

#=============== CUSTOM PROMPT TEMPLATE ===============
qa_template = """
You are an expert assistant providing detailed, complete, and accurate answers.
Question: {question}
Context: {context}
Instructions:
- If the question asks for a specific number of features, points, or reasons, always list **exactly that number**.
- Include all relevant details from the given context.
- Use clear numbering or bullet points (e.g., 1., 2., 3.).
- Do not summarize or skip any point.
Answer:
"""

qa_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=qa_template
)

#=============== GEMINI LLM + CONVERSATION CHAIN ===============
def create_conversation_chain(vectorstore):
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash-exp",
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=0.2,
        max_output_tokens=2048  # Increased for full responses
    )

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5}
        ),
        memory=memory,
        combine_docs_chain_kwargs={"prompt": qa_prompt},
        verbose=True
    )
    return chain

#=============== CHAT HANDLER ===============
def handle_user_query(user_question):
    if st.session_state.conversation is None:
        st.error("Please process documents first.")
        return

    try:
        response = st.session_state.conversation({"question": user_question})
        st.session_state.chat_history = response["chat_history"]

        # Display the full chat history
        for i, msg in enumerate(st.session_state.chat_history):
            template = user_template if i % 2 == 0 else bot_template
            st.write(template.replace("{{MSG}}", msg.content), unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Error: {str(e)}")

def clear_chat():
    st.session_state.chat_history = []
    if "conversation" in st.session_state and st.session_state.conversation:
        st.session_state.conversation.memory.clear()
    st.rerun()

#=============== STREAMLIT MAIN APP (Corrected) ===============
def main():
    st.set_page_config(page_title="Chat with PDFs (Gemini + Qdrant)", page_icon="💬", layout="wide")
    st.write(css, unsafe_allow_html=True)
    st.markdown(header_html, unsafe_allow_html=True)
    st.markdown(stats_html, unsafe_allow_html=True)

    # --- Initialize session state ---
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # --- Header Section ---
    st.markdown("""
        <div style="text-align:center; padding: 1rem 0;">
            <h1 style="color:#3b82f6; font-family:'Poppins',sans-serif;">💬 Resume Screening Assistant</h1>
            <p style="color:#94a3b8; font-size:0.95rem;">Chat with your uploaded resumes or PDFs, powered by Gemini + Qdrant</p>
        </div>
    """, unsafe_allow_html=True)

    # --- Chat Display Window ---
    st.markdown("<div class='chat-window'>", unsafe_allow_html=True)
    if st.session_state.chat_history:
        for i, msg in enumerate(st.session_state.chat_history):
            template = user_template if i % 2 == 0 else bot_template
            st.write(template.replace("{{MSG}}", msg.content), unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # --- Input Field & Send Button ---
    st.markdown("""
        <style>
            /* ... [Your existing CSS styles] ... */
        </style>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([5, 1])
    with col1:
        user_question = st.text_input("Type your question here...")
    with col2:
        if st.button("🚀 Send"):
            if user_question:
                handle_user_query(user_question)

    # --- Clear Chat Button ---
    if st.button("🧹 Clear Chat", key="clear_chat"):
        clear_chat()

    # --- Sidebar: PDF Upload & Processing ---
    with st.sidebar:
        st.subheader("📂 Upload Documents")
        
        pdf_docs = st.file_uploader(
            "Upload PDF resumes here", 
            accept_multiple_files=True, 
            type="pdf"
        )
        
        # --- Job Description uploader (COMMENTED OUT) ---
        # job_desc_file = st.file_uploader(
        #     "Upload Job Description file", 
        #     accept_multiple_files=False, 
        #     type="txt"
        # )
        # -----------------------------------------------

        if st.button("⚙️ Process Documents"):
            
            # --- Check if PDFs are uploaded ---
            if not pdf_docs: # Removed check for job_desc_file
                st.warning("Please upload at least one PDF resume.") # Updated warning
                return  # Stop if files are missing

            with st.spinner("Processing and indexing documents..."):
                try:
                    # 1. Save uploaded files to disk
                    save_uploaded_files(pdf_docs) # Removed job_desc_file
                    
                    # 2. Run resume processing (from model.py)
                    cmd1 = (
                        f"python {PROCESS_SCRIPT} {RESUME_PDF_FOLDER} "
                        f"--json_out {RESUME_CHUNKS_OUT} "
                        # f"--job_file {JOB_DESCRIPTION_FILE} " # COMMENTED OUT
                        # f"--job_out {JD_CHUNKS_OUT}" # COMMENTED OUT
                    )
                    run_command(cmd1)

                    # 3. Run embeddings (from model.py)
                    cmd2 = f"python {EMBEDDING_SCRIPT}"
                    run_command(cmd2)

                    # 4. Connect to the now-populated Qdrant store
                    vectorstore = get_qdrant_vectorstore()
                    
                    # 5. Create the conversation chain
                    st.session_state.conversation = create_conversation_chain(vectorstore)
                    
                    st.success("✅ Documents processed and loaded successfully. You can now chat!")

                except Exception as e:
                    st.error(f"Error during processing: {str(e)}")

        st.markdown(footer_html, unsafe_allow_html=True)

# --- This must be at the end, with NO indentation ---
if __name__ == "__main__":
    main()