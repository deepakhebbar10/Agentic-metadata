#!/usr/bin/env python3
import os
import ssl
import json
import psycopg2
import google.generativeai as genai
import ollama
from langchain.schema import Document
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# ── TLS / protobuf hack (Windows) ───────────────────────────────────────────────
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
# your sync‐only Postgres URI
SYNC_DB_URI = (
    f"postgresql://{DB_CONFIG['user']}:{DB_CONFIG['password']}@"
    f"{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['dbname']}"
)
# where Chroma will persist its on-disk index
PERSIST_DIR     = "./chroma_all"
COLLECTION_NAME = "all_metadata"

# ── Gemini helper ───────────────────────────────────────────────────────────────
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

# ── LLM wrapper with fallback ──────────────────────────────────────────────────
class AgentLLM:
    def __init__(self, use_gemini: bool = True):
        self.use_gemini = use_gemini

    def analyze(self, prompt: str) -> str:
        if self.use_gemini:
            try:
                return generate_with_gemini(prompt)
            except Exception as e:
                print(f"[WARN] Gemini failed, falling back to Ollama: {e}")
        # final fallback → Mistral via Ollama
        resp = ollama.chat("mistral", messages=[{"role":"user","content":prompt}])
        return resp.get("message", str(resp))

# ── Embeddings & Vector store ─────────────────────────────────────────────────
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def fetch_all(table: str, cols: tuple) -> list:
    sql = f"SELECT {', '.join(cols)} FROM {table};"
    with psycopg2.connect(SYNC_DB_URI) as conn, conn.cursor() as cur:
        cur.execute(sql)
        return cur.fetchall()

def build_vector_store():
    """
    1) Read all three tables from Postgres
    2) Turn each row into a Document
    3) (Re-)build a Chroma index on disk
    """
    os.makedirs(PERSIST_DIR, exist_ok=True)

    docs = []
    # raw metadata
    for src, meta in fetch_all("metadata_langchain_table", ("source","metadata")):
        docs.append(Document(
            page_content=json.dumps(meta),
            metadata={"source":src,"table":"raw"}
        ))

    # enriched metadata
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

    # rebuild entire index (so any new rows appear immediately)
    Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=PERSIST_DIR,
        collection_name=COLLECTION_NAME
    )
    print(f"[INFO] Indexed {len(docs)} documents into '{COLLECTION_NAME}'.")

# ── RAG: retrieve + generate ───────────────────────────────────────────────────
def rag_answer(query: str, k: int = 5, use_gemini: bool = True) -> str:
    # load the on-disk Chroma collection
    chroma = Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=PERSIST_DIR,
        embedding_function=embeddings
    )
    # retrieve top-k docs
    hits = chroma.similarity_search(query, k=k)
    if not hits:
        return "No relevant metadata found."

    # assemble a context block
    context_block = "\n\n".join(
        f"[{d.metadata['table']}] {d.metadata['source']}:\n{d.page_content}"
        for d in hits
    )

    prompt = (
        "You are an expert assistant.  Use the following metadata snippets "
        "to help answer the question.\n\n"
        f"---\n{context_block}\n---\n\n"
        f"Question: {query}\n"
        "Answer:"
    )

    llm = AgentLLM(use_gemini=use_gemini)
    return llm.analyze(prompt)

# ── REPL ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("[INFO] (Re)building vector store…")
    build_vector_store()

    print("[INFO] Ready for RAG-powered Q&A.  Type ‘exit’ to quit.")
    while True:
        q = input("\n🔍 Your question> ").strip()
        if q.lower() in ("", "exit", "quit"):
            break
        a = rag_answer(q, k=5, use_gemini=True)
        clean = a.replace("\\_", "_")
        print(f"\n🤖 Answer:\n{clean}\n")

    print("[INFO] Goodbye.")
