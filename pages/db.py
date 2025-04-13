import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_unstructured import UnstructuredLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain_core.documents import Document
from textblob import TextBlob
import re
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv

load_dotenv()

MODEL = "gemini-2.0-flash"
EMBEDDING_MODEL = "models/embedding-001"
PINECONE_INDEX_NAME = "simple-rag"
DOC_PATH = "./data/Arlington_Budget_2025.docx"
FILE_NAME = os.path.basename(DOC_PATH)

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
os.environ["PINECONE_ENVIRONMENT"] = "us-east-1"  # Add your Pinecone environment here
os.environ["STREAMLIT_WATCH_USE_POLLING"] = "true"

def ingest_pdf(DOC_PATH):
    if DOC_PATH:
        loader = UnstructuredLoader(file_path=DOC_PATH, mode="elements")
        data = loader.load()
        return data
    else:
        print("File not found")
        return None


#Splitting into chunks
def split_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=100)
    chunks = text_splitter.split_documents(documents)
    print('done splitting')
    return chunks


@st.cache_resource
def get_vector_db(uploaded_files):
    from tempfile import NamedTemporaryFile

    # Initialize Pinecone with the new API
    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    
    # Check if index exists, if not create it
    if PINECONE_INDEX_NAME not in pc.list_indexes().names():
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=768,  # Dimension for Google's embedding model
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-west-2"  # Choose appropriate region
            )
        )
    
    # Get the index
    index = pc.Index(PINECONE_INDEX_NAME)
    
    embedding = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)
    
    # Initialize the Pinecone vector store
    vector_db = PineconeVectorStore(
        index=index,
        embedding=embedding,
        text_key="text"  # Pinecone needs to know which field contains the text
    )
    
    # Get existing sources to avoid duplicates - this would need to be implemented properly
    existing_sources = set()
    try:
        # In a production app, you might want to implement a way to check for existing documents
        # This could be a separate metadata store or a query to Pinecone with appropriate filters
        pass
    except Exception as e:
        st.warning(f"Could not retrieve existing documents: {e}")

    all_clean_chunks = []

    for uploaded_file in uploaded_files:
        file_name = uploaded_file.name

        if file_name in existing_sources:
            st.warning(f"{file_name} already exists. Skipping.")
            continue

        # Save uploaded file temporarily
        with NamedTemporaryFile(delete=False, suffix=".docx") as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_file_path = tmp_file.name

        try:
            data = ingest_pdf(tmp_file_path)
            chunks = split_documents(data)

            for doc in chunks:
                if hasattr(doc, 'page_content') and hasattr(doc, 'metadata'):
                    safe_metadata = {
                        k: v for k, v in doc.metadata.items()
                        if isinstance(v, (str, int, float, bool, type(None)))
                    }
                    safe_metadata["source"] = file_name
                    all_clean_chunks.append(Document(page_content=doc.page_content, metadata=safe_metadata))
        except Exception as e:
            st.error(f"Error processing {file_name}: {e}")
            continue
        finally:
            # Clean up the temporary file
            if os.path.exists(tmp_file_path):
                os.unlink(tmp_file_path)

    if all_clean_chunks:
        vector_db.add_documents(all_clean_chunks)
        st.success(f"Uploaded {len(all_clean_chunks)} chunks from {len(uploaded_files)} files.")
    else:
        st.info("No new documents were added.")

    return vector_db


def main():
    llm = ChatGoogleGenerativeAI(model=MODEL)

    st.title("Upload City Documents to Vector DB")

    uploaded_files = st.file_uploader(
        "Upload DOCX files",
        type=["docx","pdf","txt"],
        accept_multiple_files=True
    )

    if uploaded_files:
        if st.button("Ingest to Vector DB"):
            get_vector_db(uploaded_files)
    
if __name__ == "__main__":
    main()
