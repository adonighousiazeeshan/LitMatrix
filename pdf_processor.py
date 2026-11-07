# Handles PDF loading, text splitting, and vector embedding
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import streamlit as st
from pypdf import PdfReader

load_dotenv()
@st.cache_resource
def get_embedding_model():
    return HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

UPLOAD_DIR = "./temp_pdfs"
#CHROMA_DIR = "./chroma_db"

os.makedirs(UPLOAD_DIR, exist_ok=True)


def process_uploaded_files(uploaded_files):
    """Loads PDFs, splits text, creates embeddings,
    and returns a Chroma vector store.
    """


    all_documents = []
    uploaded_filenames = []

    for uploaded_file in uploaded_files:

        if not uploaded_file.name.lower().endswith(".pdf"):
            st.warning(f"{uploaded_file.name} is not a PDF. Skipping it.")
            continue

        if uploaded_file.size == 0:
            st.warning(f"{uploaded_file.name} is empty. Skipping it.")
            continue

        MAX_FILE_SIZE = 10 * 1024 * 1024

        if uploaded_file.size > MAX_FILE_SIZE:
            st.warning(f"{uploaded_file.name} exceeds the 10 MB limit. Skipping it.")
            continue

        # Create file path BEFORE using it
        file_path = os.path.join(UPLOAD_DIR, uploaded_file.name)

        try:
            # STEP 1: Save uploaded PDF
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            # STEP 2: Validate actual PDF
            reader = PdfReader(file_path)

            if len(reader.pages) == 0:
                raise ValueError("PDF contains no pages")

            # STEP 3: Load PDF
            loader = PyPDFLoader(file_path)
            docs = loader.load()

            # STEP 4: Add filename to metadata
            for doc in docs:
                doc.metadata["filename"] = uploaded_file.name

            all_documents.extend(docs)
            uploaded_filenames.append(uploaded_file.name)

        except Exception as e:
            st.warning(
                f"Could not process '{uploaded_file.name}'. "
                f"Error: {e}"
            )

        finally:
            # STEP 5: Delete temporary PDF
            if os.path.exists(file_path):
                os.remove(file_path)


    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunked_docs = text_splitter.split_documents(all_documents)

    # Create embeddings
    try:
     
     embedding_model = get_embedding_model()

    except Exception as e:
        st.error(f"Failed to load embedding model: {e}")
        return None, []

    # Create Chroma vector store
    try:
        vector_store = Chroma.from_documents(
            documents=chunked_docs,
            embedding=embedding_model,
        )

    except Exception as e:
        st.error(f"Failed to create vector store: {e}")
        return None, []
    return vector_store, uploaded_filenames