"""
app.py - Streamlit UI
Company Knowledge Base | Amazon Internship Project
"""

import sys
import os
import streamlit as st

# Add retrieval folder to path so we can import retriever
sys.path.append(os.path.join(os.path.dirname(__file__), "retrieval"))
from retriever import rag_query

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Company Knowledge Base",
    page_icon="🏢",
    layout="centered"
)

# ── Header ────────────────────────────────────────────────────────────────────
st.title("🏢 Company Knowledge Base")
st.markdown("Ask any question about your company documents and get grounded answers with sources.")
st.divider()

# ── Cache the embedding model so it loads only once ───────────────────────────
@st.cache_resource
def load_retriever():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("all-MiniLM-L6-v2")

load_retriever()

# ── Query Input ───────────────────────────────────────────────────────────────
query = st.text_input(
    "💬 Ask a question:",
    placeholder="e.g. What is IT governance?"
)

col1, col2 = st.columns([1, 5])
with col1:
    search_clicked = st.button("🔍 Search", use_container_width=True)

# ── Results ───────────────────────────────────────────────────────────────────
if search_clicked and query.strip():
    with st.spinner("Searching knowledge base..."):
        result = rag_query(query)

    # Answer
    st.markdown("### 💬 Answer")
    st.success(result["answer"])

    # Sources
    st.markdown("### 📚 Sources Used")
    for i, chunk in enumerate(result["chunks"], 1):
        with st.expander(f"[{i}] Page {chunk['page_number']} — Similarity: {chunk['similarity']}"):
            st.markdown(f"**Source:** {chunk['source']}")
            st.markdown(f"**Page:** {chunk['page_number']}")
            st.markdown(f"**Similarity Score:** {chunk['similarity']}")
            st.divider()
            st.markdown(chunk["content"])

elif search_clicked and not query.strip():
    st.warning("Please enter a question first!")

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption("Built with LangChain, pgvector, sentence-transformers & OpenAI | Amazon Internship Project")
