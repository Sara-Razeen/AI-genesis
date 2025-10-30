# import streamlit as st
# from dotenv import load_dotenv
# from PyPDF2 import PdfReader
# from langchain.text_splitter import CharacterTextSplitter
# from langchain_google_genai import GoogleGenerativeAIEmbeddings
# from langchain_community.vectorstores import Pinecone
# from pinecone import Pinecone as PineconeClient
# from pinecone import Pinecone, ServerlessSpec
# from langchain.memory import ConversationBufferMemory
# from langchain.chains import ConversationalRetrievalChain
# from langchain_groq import ChatGroq

# import os

# load_dotenv()

# # extract text from PDF files
# def get_pdf_text(pdf_docs):
#     text = ""
#     for pdf in pdf_docs:
#         pdf_reader = PdfReader(pdf)
#         for page in pdf_reader.pages:
#             text += page.extract_text()
#     return text

# #chunking the text
# def get_text_chunks(text):
#     text_splitter = CharacterTextSplitter(
#         separator="\n",
#         chunk_size=1000,
#         chunk_overlap=200,
#         length_function=len
#     )
#     chunks = text_splitter.split_text(text)
#     return chunks

# # def get_vectorstore(text_chunks):
# #     embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={"device": "cpu"})
# #     vector_store = Pinecone.from_texts(texts=text_chunks, embedding=embeddings)
# #     return vector_store

# def get_vectorstore(text_chunks):
#     embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
#     vector_store = Pinecone.from_texts(texts=text_chunks, embeddings=embeddings, index_name="your-pinecone-index-name")
#     return vector_store


# def get_conversation_chain(vector_store) :
#     memory=ConversationBufferMemory(memory_key="chat_history", return_messages=True)
#     llm = ChatGroq(model="llama-3.3-70b-versatile")
#     conversation_chain = ConversationalRetrievalChain.from_llm(
#         llm,
#         retriever=vector_store.as_retriever()
#     )
#     return conversation_chain
# def main():
#     st.set_page_config(page_title="Chatbot Interface", page_icon="🤖")
#     st.title("🤖 Chatbot Interface")
#     st.header("Welcome to the Chatbot Application")
#     st.text_input("Ask question about your bills here...")

#     with st.sidebar:
#         st.subheader("upload your bills here")
#         pdf_docs = st.file_uploader("upload and click on process", type=["pdf", "jpg", "jpeg", "png"], accept_multiple_files=True)
#         if st.button("Process"):
#             with st.spinner("Processing..."):
#             # pdf text
#              raw_text = get_pdf_text(pdf_docs)
#             # get the chunks
#              text_chunks = get_text_chunks(raw_text)
#              st.write("Text chunks created:")
#             # create vector store
#              vector_store = get_vectorstore(text_chunks)
#             #create conversation chain
#              conversation = get_conversation_chain(vector_store) 

# if __name__ == "__main__":
#     main()



# -------------------------- Import Required Libraries --------------------------
import streamlit as st
from dotenv import load_dotenv                      
from PyPDF2 import PdfReader                        
from langchain_text_splitters import CharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings  
from langchain_community.vectorstores import Pinecone           
from pinecone import Pinecone, ServerlessSpec                 
# from langchain.chains import ConversationalRetrievalChain
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.memory import ConversationBufferMemory        
from langchain_groq import ChatGroq  
from htmltemplate import css, bot_template, user_template                            
import os                                                       

# -------------------------- Load Environment Variables --------------------------
load_dotenv() 

def get_pdf_text(pdf_docs):
    
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text() or "" 
    return text

# -------------------------- Split Extracted Text into Chunks --------------------------
def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,       
        chunk_overlap=200,  
        length_function=len
    )
    return text_splitter.split_text(text)

# -------------------------- Create Vector Store in Pinecone --------------------------
def get_vectorstore(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))                     
    index_name = "your-pinecone-index-name"                                

    vector_store = Pinecone.from_texts(
        texts=text_chunks,
        embedding=embeddings,
        index_name=index_name
    )
    return vector_store

# -------------------------- Create Conversation Chain --------------------------
# def get_conversation_chain(vector_store):
#     memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
#     llm = ChatGroq(model="llama-3.3-70b-versatile")

#     # conversation_chain = ConversationalRetrievalChain.from_llm(
#     conversation_chain = create_retrieval_chain.llm(
#         llm=llm,
#         retriever=vector_store.as_retriever(),
#         memory=memory
#     )
#     return conversation_chain

def get_conversation_chain(vector_store):
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    llm = ChatGroq(model="llama-3.3-70b-versatile")

    # Create document chain (defines how retrieved docs are combined)
    combine_docs_chain = create_stuff_documents_chain(llm)

    # Create retrieval chain
    retrieval_chain = create_retrieval_chain(
        retriever=vector_store.as_retriever(),
        combine_docs_chain=combine_docs_chain
    )

    return retrieval_chain



# -------------------------- Answer question --------------------------

def handle_user_question(user_question):
    response = st.session_state.conversation({"question": user_question})
    st.session_state.chat_history = response["chat_history"]


    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
    st.write(response)




# -------------------------- Streamlit Frontend ---------------------------------------------------
def main():
    st.set_page_config(page_title="Chatbot Interface", page_icon="🤖")
    st.title("🤖 Chatbot Interface")
    st.header("Welcome to the Chatbot Application")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

# -------------------------- Input --------------------------
    st.header("chat with your pdf")
    user_question = st.text_input("Ask a question about your bills here...")
    if user_question:
        handle_user_question(user_question)

    st.write(user_template.replace("{{MSG}}", user_question), unsafe_allow_html=True)
    st.write(bot_template.replace("{{MSG}}", "Processing your question..."), unsafe_allow_html=True)

    # Sidebar for file upload and processing
    with st.sidebar:
        st.subheader("Upload your bills here")
        pdf_docs = st.file_uploader(
            "Upload and click on Process",
            type=["pdf"],  
            accept_multiple_files=True
        )

        # Once files are uploaded, process them
        if st.button("Process") and pdf_docs:
            with st.spinner("Processing..."):
                # Step 1: Extract text from PDFs
                raw_text = get_pdf_text(pdf_docs)

                # Step 2: Split into text chunks
                text_chunks = get_text_chunks(raw_text)
                st.success("Text chunks created ✅")

                # Step 3: Create Pinecone vector store
                vector_store = get_vectorstore(text_chunks)

                # Step 4: Initialize conversation chain with LLM and retriever
                conversation = get_conversation_chain(vector_store)

                # Store the conversation in Streamlit session state
                st.session_state.conversation = conversation

            st.session_state.conversation

# -------------------------- Entry Point --------------------------
if __name__ == "__main__":
    main()
