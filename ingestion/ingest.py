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
            for page_num, page in enumerate(reader.pages, 1):
                text = page.extract_text()
                if text:
                    docs.append({
                        "text": text,
                        "source": file,
                        "page_number": page_num
                    })
    print(f"✅ Loaded {len(docs)} pages total")
    return docs

# STEP 2 - Break pages into smaller chunks
def chunk_documents(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = []
    for doc in docs:
        split_texts = splitter.split_text(doc["text"])
        for i, chunk_text in enumerate(split_texts):
            chunks.append({
                "text":        chunk_text,
                "source":      doc["source"],
                "page_number": doc["page_number"],
                "chunk_index": i
            })
    print(f"✅ Created {len(chunks)} chunks")
    return chunks

# STEP 3 - Convert chunks into numbers (embeddings)
def embed_chunks(chunks):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    texts = [chunk["text"] for chunk in chunks]
    embeddings = model.encode(texts, show_progress_bar=True)
    print(f"✅ Generated {len(embeddings)} embeddings")
    return chunks, embeddings

# STEP 4 - Store everything in the database
def store_in_db(chunks, embeddings):
    conn = psycopg2.connect(
        host=os.getenv("DB_HOST"),
        dbname=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD")
    )
    register_vector(conn)
    cur = conn.cursor()

    # Drop old table and create new one with metadata
    cur.execute("DROP TABLE IF EXISTS documents;")
    cur.execute("""
        CREATE TABLE documents (
            id          SERIAL PRIMARY KEY,
            source      TEXT,
            page_number INTEGER,
            chunk_index INTEGER,
            content     TEXT,
            embedding   vector(384)
        );
    """)

    for chunk, embedding in zip(chunks, embeddings):
        cur.execute(
            """INSERT INTO documents 
               (source, page_number, chunk_index, content, embedding) 
               VALUES (%s, %s, %s, %s, %s)""",
            (
                chunk["source"],
                chunk["page_number"],
                chunk["chunk_index"],
                chunk["text"],
                embedding.tolist()
            )
        )

    conn.commit()
    cur.close()
    conn.close()
    print("✅ All chunks stored in database with metadata!")

# RUN EVERYTHING
if __name__ == "__main__":
    docs = load_documents("data/")
    chunks = chunk_documents(docs)
    chunks, embeddings = embed_chunks(chunks)
    store_in_db(chunks, embeddings)
