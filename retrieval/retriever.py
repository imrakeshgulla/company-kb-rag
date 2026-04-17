"""
retriever.py - RAG Retrieval Pipeline
Company Knowledge Base | Amazon Internship Project
"""

import os
import psycopg2
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────
DB_CONFIG = {
    "host":   os.getenv("DB_HOST"),
    "dbname": os.getenv("DB_NAME"),
    "user":   os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
}

EMBED_MODEL = "all-MiniLM-L6-v2"
TOP_K       = 5
GPT_MODEL   = "gpt-4o-mini"


# ── Helpers ───────────────────────────────────────────────────────────────────
def get_connection():
    return psycopg2.connect(**DB_CONFIG)


def embed_query(query: str, model: SentenceTransformer) -> list:
    vector = model.encode(query, normalize_embeddings=True)
    return vector.tolist()


def retrieve_chunks(query_vec: list, top_k: int = TOP_K) -> list:
    conn = get_connection()
    cur  = conn.cursor()

    sql = """
        SELECT
            id,
            source,
            page_number,
            chunk_index,
            content,
            1 - (embedding <=> %s::vector) AS similarity
        FROM documents
        ORDER BY embedding <=> %s::vector
        LIMIT %s;
    """

    vec_str = "[" + ",".join(map(str, query_vec)) + "]"
    cur.execute(sql, (vec_str, vec_str, top_k))
    rows = cur.fetchall()
    cur.close()
    conn.close()

    chunks = []
    for row in rows:
        chunks.append({
            "id":          row[0],
            "source":      row[1],
            "page_number": row[2],
            "chunk_index": row[3],
            "content":     row[4],
            "similarity":  round(float(row[5]), 4),
        })
    return chunks


def build_context(chunks: list) -> str:
    parts = []
    for i, c in enumerate(chunks, 1):
        parts.append(
            f"[{i}] Source: {c['source']} | Page {c['page_number']}\n{c['content']}"
        )
    return "\n\n---\n\n".join(parts)


def generate_answer(query: str, context: str, client: OpenAI) -> str:
    system_prompt = (
        "You are a precise knowledge-base assistant. "
        "Answer ONLY using the provided context. "
        "If the answer is not in the context, say 'I don't have enough information.' "
        "Cite the source and page number when referencing specific facts."
    )

    user_prompt = f"""Context:
{context}

Question: {query}

Answer:"""

    response = client.chat.completions.create(
        model=GPT_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
        temperature=0.2,
        max_tokens=600,
    )
    return response.choices[0].message.content.strip()


def rag_query(query: str, top_k: int = TOP_K, verbose: bool = False) -> dict:
    embed_model   = SentenceTransformer(EMBED_MODEL)
    openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    query_vec = embed_query(query, embed_model)
    chunks    = retrieve_chunks(query_vec, top_k)

    if verbose:
        print(f"\n🔍 Query: {query}")
        print(f"📦 Retrieved {len(chunks)} chunks:")
        for c in chunks:
            print(f"  [{c['similarity']:.4f}] {c['source']} p.{c['page_number']}")

    if not chunks:
        return {"query": query, "answer": "No relevant documents found.", "chunks": []}

    context = build_context(chunks)
    answer  = generate_answer(query, context, openai_client)

    return {"query": query, "answer": answer, "chunks": chunks}


# ── CLI test ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "What is IT strategy?"

    result = rag_query(query, verbose=True)

    print("\n" + "="*60)
    print("💬 ANSWER")
    print("="*60)
    print(result["answer"])
    print("\n📚 Sources used:")
    for c in result["chunks"]:
        print(f"  • {c['source']} — page {c['page_number']} (similarity {c['similarity']})")
