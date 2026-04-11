import os
from dotenv import load_dotenv
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import psycopg2
from pgvector.psycopg2 import register_vector

load_dotenv()

# STEP 1 - Read PDF files from the data folder
def load_documents(folder_path):
    docs = []
    for file in os.listdir(folder_path):
        if file.endswith(".pdf"):
            print(f"📄 Reading: {file}")
            reader = PdfReader(os.path.join(folder_path, file))
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    docs.append(text)
    print(f"✅ Loaded {len(docs)} pages total")
    return docs

# STEP 2 - Break pages into smaller chunks
def chunk_documents(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.create_documents(docs)
    print(f"✅ Created {len(chunks)} chunks")
    return chunks

# STEP 3 - Convert chunks into numbers (embeddings)
def embed_chunks(chunks):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    texts = [chunk.page_content for chunk in chunks]
    embeddings = model.encode(texts, show_progress_bar=True)
    print(f"✅ Generated {len(embeddings)} embeddings")
    return texts, embeddings

# STEP 4 - Store everything in the database
def store_in_db(texts, embeddings):
    conn = psycopg2.connect(
        host=os.getenv("DB_HOST"),
        dbname=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD")
    )
    register_vector(conn)
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            id SERIAL PRIMARY KEY,
            content TEXT,
            embedding vector(384)
        );
    """)

    for text, embedding in zip(texts, embeddings):
        cur.execute(
            "INSERT INTO documents (content, embedding) VALUES (%s, %s)",
            (text, embedding.tolist())
        )

    conn.commit()
    cur.close()
    conn.close()
    print("✅ All chunks stored in database!")

# RUN EVERYTHING
if __name__ == "__main__":
    docs = load_documents("data/")
    chunks = chunk_documents(docs)
    texts, embeddings = embed_chunks(chunks)
    store_in_db(texts, embeddings)