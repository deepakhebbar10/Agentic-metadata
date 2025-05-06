#!/usr/bin/env python3
import os
import ssl
import json
import psycopg2
import streamlit as st

import google.generativeai as genai
import ollama

from langchain.schema import Document
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# ── TLS / protobuf hack for Windows ─────────────────────────────────────────────
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
ssl._create_default_https_context = ssl._create_unverified_context

# ── Configuration ───────────────────────────────────────────────────────────────
DB_CONFIG = {
    "dbname":   "metadata",
    "user":     "postgres",
    "password": "deepak",
    "host":     "localhost",
    "port":     "5432",
}
SYNC_DB_URI    = (
    f"postgresql://{DB_CONFIG['user']}:{DB_CONFIG['password']}@"
    f"{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['dbname']}"
)
PERSIST_DIR     = "./chroma_all"
COLLECTION_NAME = "all_metadata"

GEMINI_API_KEY_FILE = "gemini_api_key.txt"
GEMINI_MODEL        = "gemini-2.0-flash"

def load_gemini_api_key() -> str:
    here = os.path.dirname(__file__)
    return open(os.path.join(here, GEMINI_API_KEY_FILE)).read().strip()

def generate_with_gemini(prompt: str) -> str:
    genai.configure(api_key=load_gemini_api_key())
    model = genai.GenerativeModel(GEMINI_MODEL)
    resp  = model.generate_content(prompt)
    return resp.text if hasattr(resp, "text") else str(resp)

class AgentLLM:
    def __init__(self, use_gemini: bool = True):
        self.use_gemini = use_gemini

    def analyze(self, prompt: str) -> str:
        if self.use_gemini:
            try:
                return generate_with_gemini(prompt)
            except Exception as e:
                print(f"[WARN] Gemini failed, falling back to Ollama: {e}")
        resp = ollama.chat("mistral", messages=[{"role":"user","content":prompt}])
        return resp.get("message", str(resp))

# ── Embeddings & Vector store ─────────────────────────────────────────────────
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def fetch_all(table: str, cols: tuple) -> list:
    sql = f"SELECT {', '.join(cols)} FROM {table};"
    with psycopg2.connect(SYNC_DB_URI) as conn, conn.cursor() as cur:
        cur.execute(sql)
        return cur.fetchall()

@st.cache_resource(show_spinner=False)
def build_or_load_index():
    """
    Build Chroma index if not already present, otherwise reopen it.
    """
    os.makedirs(PERSIST_DIR, exist_ok=True)
    index_path = os.path.join(PERSIST_DIR, COLLECTION_NAME)
    # if empty or missing, rebuild
    if not os.path.isdir(index_path) or not os.listdir(index_path):
        docs = []
        # raw metadata
        for src, meta in fetch_all("metadata_langchain_table", ("source","metadata")):
            docs.append(Document(
                page_content=json.dumps(meta),
                metadata={"source":src,"table":"raw"}
            ))
        # enriched
        for src, desc in fetch_all("metadata_langchain_enriched",
                                   ("source","enriched_description")):
            docs.append(Document(
                page_content=desc,
                metadata={"source":src,"table":"enriched"}
            ))
        # relationships
        for src, rel in fetch_all("metadata_langchain_relationships",
                                  ("source","relationships")):
            text = rel if isinstance(rel, str) else json.dumps(rel)
            docs.append(Document(
                page_content=text,
                metadata={"source":src,"table":"relationships"}
            ))

        Chroma.from_documents(
            documents=docs,
            embedding=embeddings,
            persist_directory=PERSIST_DIR,
            collection_name=COLLECTION_NAME
        )

    # reopen
    return Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=PERSIST_DIR,
        embedding_function=embeddings
    )

def rag_answer(query: str, k: int = 5, use_gemini: bool = True) -> str:
    chroma = build_or_load_index()
    hits = chroma.similarity_search(query, k=k)
    if not hits:
        return "No relevant metadata found."

    # build context
    context = "\n\n".join(
        f"[{d.metadata['table']}] {d.metadata['source']}:\n{d.page_content}"
        for d in hits
    )
    prompt = (
        "You are an expert AI assistant.  Use the following metadata snippets "
        "to answer the question as accurately as possible.\n\n"
        f"---\n{context}\n---\n\n"
        f"Question: {query}\nAnswer:"
    )

    llm = AgentLLM(use_gemini=use_gemini)
    answer = llm.analyze(prompt)
    # remove markdown escapes of underscores
    return answer.replace("\\_", "_")

# ── Streamlit UI ────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Metadata RAG Search", layout="wide")
st.title("📚 RAG-powered Metadata Q&A")

st.markdown("""
Enter any natural-language question about your metadata.  
I’ll retrieve the most relevant snippets and use Gemini (or Mistral) to fuse them into an answer.
""")

query = st.text_input("🔍 Your question", "")
use_gemini = st.checkbox("Use Google Gemini (otherwise Mistral via Ollama)", value=True)

if st.button("Search"):
    if not query:
        st.warning("Please type a question above to search.")
    else:
        with st.spinner("Retrieving & generating answer…"):
            result = rag_answer(query, k=5, use_gemini=use_gemini)
        st.subheader("🤖 Answer")
        st.write(result)
        st.markdown("---")
        st.subheader("📋 Retrieved snippets")
        chroma = build_or_load_index()
        hits = chroma.similarity_search(query, k=5)
        for i, doc in enumerate(hits, 1):
            st.markdown(f"**{i}. [{doc.metadata['table']}] {doc.metadata['source']}**")
            st.code(doc.page_content[:300] + ("…" if len(doc.page_content) > 300 else ""))
            st.markdown("---")
