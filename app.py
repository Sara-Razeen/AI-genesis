
from typing import List
import os
import subprocess
import shutil
import streamlit as st
from dotenv import load_dotenv, find_dotenv
from typing import List
from PyPDF2 import PdfReader

# ---------------- LangChain Imports ----------------
from langchain_community.embeddings import HuggingFaceEmbeddings
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
        model_name="BAAI/bge-base-en"
    )

    qdrant_url = os.getenv("QDRANT_URL", "https://60545997-4f65-414f-879d-4c1ee500d0c2.europe-west3-0.gcp.cloud.qdrant.io")
    qdrant_api_key = os.getenv("QDRANT_API_KEY", None)
    collection_name = os.getenv("QDRANT_COLLECTION", "cv_vectors")

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
        model="gemini-2.5-flash",
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=0.2,
        max_output_tokens=2048,
        max_retries=1
    )

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 20}
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
        with st.spinner("AI is thinking..."):
            response = st.session_state.conversation({"question": user_question})
            st.session_state.chat_history = response["chat_history"]
        st.success("Response received!")
        st.rerun()

    except Exception as e:
        st.error(f"Error: {str(e)}")
        st.error("Make sure you have:")
        st.error("1. Uploaded PDF documents")
        st.error("2. Clicked 'Process Documents'")
        st.error("3. Set up your environment variables (.env file)")

def clear_chat():
    st.session_state.chat_history = []
    if "conversation" in st.session_state and st.session_state.conversation:
        st.session_state.conversation.memory.clear()
    st.rerun()

#=============== STREAMLIT CHAT INTERFACE ===============
def render_chat_interface():
    """Render chat interface using Streamlit components"""
    
    # Custom CSS for chat styling
    st.markdown("""
    <style>
   
    .user-message {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: #f8fafc;
        padding: 1rem 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        margin-left: 3rem;
    }
    
    .bot-message {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);        
        color: #f8fafc;
        border: 1px solid #e2e8f0;
        padding: 1rem 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        margin-right: 3rem;
    }

    </style>
    """, unsafe_allow_html=True)
    
    # Chat container
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    
    # Display chat history
    if st.session_state.chat_history:
        for i, msg in enumerate(st.session_state.chat_history):
            if i % 2 == 0:  # User message
                st.markdown(f'<div class="user-message">👤 You: {msg.content}</div>', unsafe_allow_html=True)
            else:  # Bot message
                st.markdown(f'<div class="bot-message"> AI: {msg.content}</div>', unsafe_allow_html=True)
       
    
    st.markdown('</div>', unsafe_allow_html=True)

#=============== STREAMLIT MAIN APP ===============
def main():
    st.set_page_config(
        page_title="AI Resume Screening Assistant", 
        # page_icon="", 
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom styling
    hide_streamlit_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stApp {
    background: #475569 !important;  
}
    
    .stTextInput > div > div > input {
        border-radius: 40px !important;
        border: 0px solid #e2e8f0 !important;
        padding: 1rem !important;
        font-size: 1rem !important;
        background: transparent !important;
    }
    
    .stButton > button {
        background: #4b6eaf !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 0.4rem 1rem !important;
        margin-top: 30px !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4) !important;
    }
    </style>
    """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True)

    # --- Initialize session state ---
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # --- Header ---
    st.markdown("""
    <div style="text-align: center; padding: 2rem 0; margin-bottom: 2rem;">
        <h1 style="color: white; font-size: 3rem; text-shadow: 0 2px 4px rgba(0,0,0,0.1);">🤖 AI Resume Screening Assistant</h1>
        <p style="color: #e2e8f0; font-size: 1.2rem;">Intelligent candidate analysis powered by Gemini AI & Qdrant Vector Search</p>
    </div>
    """, unsafe_allow_html=True)
    
    # --- Debug info ---
    if st.session_state.conversation is not None:
        st.sidebar.success("AI System Ready!")
    else:
        st.sidebar.warning("Please upload and process documents first")
    
    # --- Chat Interface ---
    render_chat_interface()
    
    # --- Input Area ---
    col1, col2, col3 = st.columns([7, 1.5, 1.5])
    
    with col1:
        user_question = st.text_area("Ask me anything about the resumes...",height=100 , key="user_input", placeholder="e.g., Who has Python experience?")
    
    with col2:
        send_clicked = st.button("Send")
        if send_clicked and user_question.strip():
            handle_user_query(user_question)
    
    with col3:
        if st.button("Clear"):
            clear_chat()

    

    # --- Sidebar: PDF Upload & Processing ---
    with st.sidebar:
        st.subheader("Upload Documents")
        
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

        if st.button("Process Documents"):
            
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
                    
                    st.success("Documents processed and loaded successfully. You can now chat!")

                except Exception as e:
                    st.error(f"Error during processing: {str(e)}")

        st.markdown("---")
        st.markdown("Questions:")
        st.markdown("1. Who has Python skills?")
        st.markdown("2. List all candidates name?")
        # st.markdown("- Try 'Show candidates with [skill]' for targeted search")

# --- This must be at the end, with NO indentation ---
if __name__ == "__main__":
    main()