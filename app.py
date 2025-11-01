import os
import streamlit as st
from dotenv import load_dotenv, find_dotenv
from PyPDF2 import PdfReader
from typing import List

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
from htmltemplate import css, bot_template, user_template



load_dotenv(find_dotenv())  # load .env file so GOOGLE_API_KEY is available

#=============== PDF PROCESSING ===============

def extract_text_from_pdfs(pdf_docs) -> str:
    """Extract raw text from uploaded PDF files."""
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            if page.extract_text():
                text += page.extract_text()
    return text


def split_text_into_chunks(text: str) -> List[Document]:
    """Split extracted text into overlapping chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        separators=["\n", ".", "?", "!"]
    )
    chunks = splitter.split_text(text)
    return [Document(page_content=chunk) for chunk in chunks]



#=============== QDRANT VECTOR STORE ===============
def create_qdrant_vectorstore(text_chunks):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

    # Qdrant client setup
    qdrant_url = os.getenv("QDRANT_URL", "https://60545997-4f65-414f-879d-4c1ee500d0c2.europe-west3-0.gcp.cloud.qdrant.io")
    qdrant_api_key = os.getenv("QDRANT_API_KEY", None)
    collection_name = os.getenv("QDRANT_COLLECTION", "pdf_chat_collection")

    client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)

    vectorstore = Qdrant.from_documents(
        documents=text_chunks,
        embedding=embeddings,
        url=qdrant_url,
        api_key=qdrant_api_key,
        collection_name=collection_name
    )

    return vectorstore



#=============== CUSTOM PROMPT TEMPLATE (Ensures complete answers) ===============
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
    if st.session_state.conversation:
        st.session_state.conversation.memory.clear()
    st.rerun()


#=============== STREAMLIT MAIN APP ===============
def main():
    load_dotenv(find_dotenv())
    st.set_page_config(page_title="Chat with PDFs (Gemini + Qdrant)", page_icon="📚", layout="wide")

    st.write(css, unsafe_allow_html=True)

    # Initialize session states
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    st.header("💬 Resume Screening")

    # Clear chat
    col1, col2 = st.columns([3, 1])
    with col2:
        if st.button("🧹 Clear Chat"):
            clear_chat()

    # User query
    user_question = st.text_input("Ask something about your uploaded documents:")
    if user_question:
        handle_user_query(user_question)

    # Sidebar for PDF upload & processing
    with st.sidebar:
        st.subheader("📂 Upload PDFs")
        pdf_docs = st.file_uploader("Upload PDF files here", accept_multiple_files=True, type="pdf")

        if st.button("⚙️ Process Documents"):
            if not pdf_docs:
                st.warning("Please upload at least one PDF.")
                return

            with st.spinner("Processing and indexing documents..."):
                try:
                    text = extract_text_from_pdfs(pdf_docs)
                    if not text.strip():
                        st.error("No readable text found in PDFs.")
                        return

                    chunks = split_text_into_chunks(text)
                    st.info(f"Created {len(chunks)} text chunks.")

                    vectorstore = create_qdrant_vectorstore(chunks)
                    st.session_state.conversation = create_conversation_chain(vectorstore)
                    st.success(" Documents processed successfully. You can now chat!")

                except Exception as e:
                    st.error(f"Error during processing: {str(e)}")

        st.markdown("---")
        st.markdown("""
        ### ⚙️ Tech Stack
        - **Google Gemini (LLM)**
        - **Qdrant (Vector Database)**
        - **Hugging Face (Embeddings)**
        - **LangChain + Streamlit**
        """)


if __name__ == "__main__":
    main()
