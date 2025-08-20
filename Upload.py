import os
import uuid
import pandas as pd
import streamlit as st
from io import BytesIO
from sqlalchemy import create_engine
from sqlalchemy.exc import SQLAlchemyError

import PyPDF2
import docx
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer


# =========================================================
#                 CONFIGURATION & CONSTANTS
# =========================================================
DB_USER = "postgres"
DB_PASSWORD = "United2025"
DB_HOST = "localhost"
DB_PORT = "5432"
DB_NAME = "AutoMotor_Insurance"
DB_TABLE_NAME = "motor_insurance_data"

CHROMA_DIR = "vector_db"
os.makedirs(CHROMA_DIR, exist_ok=True)

EMBED_MODEL_NAME = "all-MiniLM-L6-v2"


# =========================================================
#                  CACHED RESOURCES
# =========================================================
@st.cache_resource
def get_engine():
    """Create and cache a SQLAlchemy engine."""
    try:
        engine_string = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
        engine = create_engine(engine_string)
        return engine
    except Exception as e:
        st.error(f"Error connecting to the database: {e}")
        return None


@st.cache_resource
def get_embedder():
    """Load and cache sentence transformer model."""
    return SentenceTransformer(EMBED_MODEL_NAME)


@st.cache_resource
def get_chroma_collection():
    """Get or create Chroma vector DB collection."""
    client = chromadb.PersistentClient(path=CHROMA_DIR, settings=Settings(allow_reset=False))
    return client.get_or_create_collection("documents")


# =========================================================
#                        HELPERS
# =========================================================
def extract_pdf_text(uploaded_file) -> str:
    """Extract text from a PDF file."""
    reader = PyPDF2.PdfReader(uploaded_file)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text() or ""
        text += page_text + "\n"
    return text.strip()


def extract_docx_text(uploaded_file) -> str:
    """Extract text from a DOCX file."""
    doc = docx.Document(uploaded_file)
    return "\n".join(para.text for para in doc.paragraphs if para.text.strip())


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50):
    """Split text into overlapping chunks."""
    words = text.split()
    if not words:
        return []

    chunks, start = [], 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        if end == len(words):
            break
        start = max(0, end - overlap)
    return chunks


def store_in_vector_db(raw_text: str, filename: str):
    """Embed and store text chunks into Chroma vector DB."""
    collection = get_chroma_collection()
    embedder = get_embedder()

    chunks = chunk_text(raw_text)
    if not chunks:
        raise ValueError("No text extracted from file (empty or unsupported format).")

    embeddings = embedder.encode(chunks).tolist()
    unique_prefix = f"{filename}_{uuid.uuid4().hex}"
    ids = [f"{unique_prefix}_{i}" for i in range(len(chunks))]
    metadatas = [{"filename": filename, "chunk_index": i} for i in range(len(chunks))]

    collection.add(documents=chunks, embeddings=embeddings, ids=ids, metadatas=metadatas)


# =========================================================
#                      UPLOAD PAGE
# =========================================================
def show():
    st.header("Upload Section")

    if "uploaded" not in st.session_state:
        st.session_state.uploaded = None

    uploaded_file = st.file_uploader(
        "Choose a file",
        type=["csv", "xls", "xlsx", "pdf", "docx"],
        key="file_uploader"
    )

    if uploaded_file:
        st.session_state.uploaded = uploaded_file
        st.success(f"File '{uploaded_file.name}' uploaded successfully!")

        with st.expander("File details"):
            st.write("**Name:**", uploaded_file.name)
            st.write("**Type:**", uploaded_file.type)
            st.write("**Size:**", f"{uploaded_file.size} bytes")

    if st.session_state.uploaded and st.button("Process and Save"):
        uploaded_file = st.session_state.uploaded
        try:
            # ================= CSV / Excel =================
            if uploaded_file.type in [
                "text/csv",
                "application/vnd.ms-excel",
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            ]:
                if uploaded_file.type == "text/csv":
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)

                df.columns = [str(col).strip().lower().replace(" ", "_") for col in df.columns]

                engine = get_engine()
                if engine is None:
                    st.error("Database engine not available.")
                    return

                with engine.begin() as conn:
                    with st.spinner("Saving structured data to PostgreSQL..."):
                        df.to_sql(DB_TABLE_NAME, con=conn, if_exists="append", index=False)
                st.success("Data saved to PostgreSQL successfully!")

            # ================= PDF =================
            elif uploaded_file.type == "application/pdf":
                data = uploaded_file.read()
                text = extract_pdf_text(BytesIO(data))
                store_in_vector_db(text, uploaded_file.name)
                st.success("PDF text stored in Vector DB for Q&A!")

            # ================= DOCX =================
            elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                data = uploaded_file.read()
                text = extract_docx_text(BytesIO(data))
                store_in_vector_db(text, uploaded_file.name)
                st.success("DOCX text stored in Vector DB for Q&A!")

            else:
                st.warning("Unsupported file type.")

            # ✅ Reset after processing so it doesn’t show again
            st.session_state.uploaded = None
            st.session_state.file_uploader = None

        except SQLAlchemyError as e:
            st.error(f"Database error: {e}")
        except Exception as e:
            st.error(f"Error processing file: {e}")
    