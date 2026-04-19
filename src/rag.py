import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

VECTOR_DB_DIR = "data/vectorstore"
KNOWLEDGE_BASE_PATH = "knowledge_base/lending_guidelines.txt"

def setup_vector_db():
    """
    Reads the text guidelines, splits them into manageable chunks,
    creates numerical embeddings, and saves them into a local FAISS vector database.
    """
    if os.path.exists(VECTOR_DB_DIR) and len(os.listdir(VECTOR_DB_DIR)) > 0:
        # DB already exists, skip
        return
    
    if not os.path.exists(KNOWLEDGE_BASE_PATH):
        raise FileNotFoundError(f"Knowledge base not found at {KNOWLEDGE_BASE_PATH}")

    # 1. Load Document
    loader = TextLoader(KNOWLEDGE_BASE_PATH)
    docs = loader.load()

    # 2. Split Document into smaller manageable chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    split_docs = text_splitter.split_documents(docs)

    # 3. Create Embeddings (using a free open-source HuggingFace model)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # 4. Save to Vector Store (FAISS)
    vector_db = FAISS.from_documents(split_docs, embeddings)
    vector_db.save_local(VECTOR_DB_DIR)

def get_retriever():
    """
    Loads the saved FAISS vector database and returns a retriever object.
    """
    if not os.path.exists(VECTOR_DB_DIR):
        setup_vector_db()

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_db = FAISS.load_local(VECTOR_DB_DIR, embeddings, allow_dangerous_deserialization=True)
    return vector_db.as_retriever(search_kwargs={"k": 3})
