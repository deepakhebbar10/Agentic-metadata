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
    "dbname":   "langchain_metadata",
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

# ── S3 settings ────────────────────────────────────────────────────────────────
S3_BUCKET = os.getenv("S3_BUCKET", "langchaindata")
S3_PREFIX = os.getenv("S3_PREFIX", "").strip("/")
AWS_REGION = os.getenv("AWS_REGION", "ap-south-1")

# ── Gemini / Ollama setup ──────────────────────────────────────────────────────
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
@st.cache_resource(show_spinner=False)
def get_embeddings():
    # Correct instantiation: model_name + model_kwargs; no `client=` argument
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device":"cpu"},
    )

def fetch_row(table, key_col, val_col, key):
    sql = f"SELECT {val_col} FROM {table} WHERE {key_col} = %s;"
    with psycopg2.connect(SYNC_DB_URI) as conn, conn.cursor() as cur:
        cur.execute(sql, (key,))
        r = cur.fetchone()
        return r[0] if r else None

def fetch_all(table, cols):
    sql = f"SELECT {', '.join(cols)} FROM {table};"
    with psycopg2.connect(SYNC_DB_URI) as conn, conn.cursor() as cur:
        cur.execute(sql)
        return cur.fetchall()

@st.cache_resource(show_spinner=False)
def get_chroma_index():
    embeddings = get_embeddings()
    idx_path = os.path.join(PERSIST_DIR, COLLECTION_NAME)
    if not os.path.isdir(idx_path) or not os.listdir(idx_path):
        os.makedirs(PERSIST_DIR, exist_ok=True)
        docs = []
        # raw
        for src, meta in fetch_all("metadata_langchain_table", ("source","metadata")):
            docs.append(Document(page_content=json.dumps(meta),
                                 metadata={"source":src,"table":"raw"}))
        # enriched
        for src, desc in fetch_all("metadata_langchain_enriched","source, enriched_description".split(",")):
            docs.append(Document(page_content=desc,
                                 metadata={"source":src,"table":"enriched"}))
        # relationships
        for src, rel in fetch_all("metadata_langchain_relationships","source, relationships".split(",")):
            text = rel if isinstance(rel,str) else json.dumps(rel)
            docs.append(Document(page_content=text,
                                 metadata={"source":src,"table":"relationships"}))

        # --- Add invoice table ---
        invoice_rows = fetch_all(
            "invoice",
            [
                "id", "invoice_id", "customer", "product", "category",
                "original_price", "discounted_price", "discount",
                "product_link", "context", "sentiment_json"
            ]
        )
        for row in invoice_rows:
            inv_id = row[0]
            inv_dict = {
                "invoice_id": row[1],
                "customer": row[2],
                "product": row[3],
                "category": row[4],
                "original_price": row[5],
                "discounted_price": row[6],
                "discount": row[7],
                "product_link": row[8],
                "context": row[9],
                "sentiment_json": row[10],
            }
            docs.append(Document(
                page_content=json.dumps(inv_dict),
                metadata={"source": f"invoice_{inv_id}", "table": "invoice"}
            ))

        # --- Add review table ---
        review_rows = fetch_all(
            "review",
            [
                "id", "source", "customer_name", "product_name",
                "sentiment_label", "sentiment_score", "sentiment"
            ]
        )
        for row in review_rows:
            rev_id = row[0]
            review_dict = {
                "source": row[1],
                "customer_name": row[2],
                "product_name": row[3],
                "sentiment_label": row[4],
                "sentiment_score": row[5],
                "sentiment": row[6]
            }
            docs.append(Document(
                page_content=json.dumps(review_dict),
                metadata={"source": f"review_{rev_id}", "table": "review"}
            ))

        # --- Add relationships table ---
        relationship_rows = fetch_all(
            "metadata_langchain_relationships",
            ["source", "relationships"]
        )
        for row in relationship_rows:
            src = row[0]
            rel = row[1]
            # If relationships is not a string, convert to JSON string
            rel_text = rel if isinstance(rel, str) else json.dumps(rel)
            docs.append(Document(
                page_content=rel_text,
                metadata={"source": src, "table": "relationships"}
            ))

        # --- Add Relationships table (file-to-file relationships) ---
        relationships_rows = fetch_all(
            "relationships",
            ["file1", "file2", "relationship_type", "description"]
        )
        for row in relationships_rows:
            rel_dict = {
                "file1": row[0],
                "file2": row[1],
                "relationship_type": row[2],
                "description": row[3]
            }
            docs.append(Document(
                page_content=json.dumps(rel_dict),
                metadata={
                    "source": f"relationship_{row[0]}_{row[1]}",
                    "table": "relationships"
                }
            ))

        # --- Add table_relationship table ---
        table_relationship_rows = fetch_all(
            "table_relationship",
            ["table1", "table2", "relationship", "relationship_type", "description"]
        )
        for row in table_relationship_rows:
            rel_dict = {
                "table1": row[0],
                "table2": row[1],
                "relationship": row[2],
                "relationship_type": row[3],
                "description": row[4]
            }
            docs.append(Document(
                page_content=json.dumps(rel_dict),
                metadata={
                    "source": f"table_relationship_{row[0]}_{row[1]}",
                    "table": "table_relationship"
                }
            ))

        Chroma.from_documents(
            documents=docs,
            embedding=embeddings,
            persist_directory=PERSIST_DIR,
            collection_name=COLLECTION_NAME
        )
    return Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=PERSIST_DIR,
        embedding_function=embeddings
    )

def semantic_search(query: str, k: int = 5):
    chroma = get_chroma_index()
    raw_hits = chroma.similarity_search(query, k=k*2)
    sources, seen = [], set()
    for doc in raw_hits:
        src = doc.metadata["source"]
        if src in ("DB_Columns","DB_Insights"):
            continue
        if src not in seen:
            seen.add(src)
            sources.append(src)
            if len(sources) >= k:
                break
    return sources

def rag_answer(question: str, sources: list[str], use_gemini: bool):
    pieces = []
    for src in sources:
        r   = fetch_row("metadata_langchain_table","source","metadata",src)
        e   = fetch_row("metadata_langchain_enriched","source","enriched_description",src)
        rel = fetch_row("metadata_langchain_relationships","source","relationships",src)
        pieces.append(f"[{src} RAW]\n{json.dumps(r)}")
        pieces.append(f"[{src} ENRICHED]\n{e}")
        pieces.append(f"[{src} REL]\n{json.dumps(rel)}")
    context = "\n\n".join(pieces)

    prompt = (
        "You are an expert assistant.  Use the following metadata snippets to answer the question.\n\n"
        f"---\n{context}\n---\n\n"
        f"Question: {question}\nAnswer:"
    )
    llm = AgentLLM(use_gemini=use_gemini)
    return llm.analyze(prompt).replace("\\_","_")

def get_top_invoices_with_negative_reviews(product_keyword="tv", top_n=3):
    # 1. Find invoices for smart TVs
    sql_invoice = """
        SELECT id, invoice_id, customer, product, category, original_price, discounted_price, discount, product_link, context, sentiment_json
        FROM invoice
        WHERE LOWER(product) LIKE %s
        ORDER BY discounted_price DESC
        LIMIT 20;
    """
    invoices = []
    with psycopg2.connect(SYNC_DB_URI) as conn, conn.cursor() as cur:
        cur.execute(sql_invoice, (f"%{product_keyword.lower()}%",))
        invoices = cur.fetchall()

    # 2. For each invoice, check for negative review
    results = []
    for inv in invoices:
        product_name = inv[3]
        # Find negative review for this product
        sql_review = """
            SELECT sentiment_label, sentiment_score, sentiment
            FROM review
            WHERE LOWER(product_name) = %s AND sentiment_label = 'negative'
            ORDER BY sentiment_score ASC
            LIMIT 1;
        """
        with psycopg2.connect(SYNC_DB_URI) as conn, conn.cursor() as cur:
            cur.execute(sql_review, (product_name.lower(),))
            review = cur.fetchone()
            if review:
                results.append({
                    "invoice": {
                        "invoice_id": inv[1],
                        "customer": inv[2],
                        "product": inv[3],
                        "category": inv[4],
                        "original_price": inv[5],
                        "discounted_price": inv[6],
                        "discount": inv[7],
                        "product_link": inv[8],
                        "context": inv[9],
                        "sentiment_json": inv[10],
                    },
                    "review": {
                        "sentiment_label": review[0],
                        "sentiment_score": review[1],
                        "sentiment": review[2],
                    }
                })
        if len(results) >= top_n:
            break
    return results

# ── Streamlit UI ────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Metadata RAG Search", layout="wide")
st.title("📚 RAG-powered Metadata Q&A")

st.markdown("""
Enter any natural-language question about your metadata.  
I’ll retrieve the most relevant items and use Gemini (or Mistral) to fuse them into an answer.
""")

query      = st.text_input("🔍 Your question")
use_gemini = st.checkbox("Use Gemini (otherwise Mistral)", value=True)

if st.button("Search"):
    if not query:
        st.warning("Please enter a question.")
    else:
        with st.spinner("Retrieving snippets…"):
            top_srcs = semantic_search(query, k=5)

        if not top_srcs:
            st.info("No relevant metadata found.")
        else:
            # 1️⃣ RAG answer
            with st.spinner("Generating answer…"):
                ans = rag_answer(query, top_srcs, use_gemini)
            st.subheader("🤖 Answer")
            st.write(ans)
            st.markdown("---")

            # 2️⃣ Show each source’s details + AWS Console link
            for i, src in enumerate(top_srcs, start=1):
                st.subheader(f"{i}. `{src}`")

                # if it’s a file (not DB), render AWS console link
                if src not in ("DB_Columns","DB_Insights"):
                    key = f"{S3_PREFIX}/{src}" if S3_PREFIX else src
                    console_url = (
                        f"https://{AWS_REGION}.console.aws.amazon.com/s3/object/"
                        f"{S3_BUCKET}?region={AWS_REGION}"
                        f"&bucketType=general&prefix={key}"
                    )
                    st.markdown(f"[🔗 Open in S3 Console]({console_url})")

                # --- Handle invoice and review tables ---
                if src.startswith("invoice_"):
                    inv_id = src.replace("invoice_", "")
                    sql = """
                        SELECT invoice_id, customer, product, category, original_price, discounted_price, discount, product_link, context, sentiment_json
                        FROM invoice WHERE id = %s;
                    """
                    with psycopg2.connect(SYNC_DB_URI) as conn, conn.cursor() as cur:
                        cur.execute(sql, (inv_id,))
                        row = cur.fetchone()
                        if row:
                            inv_dict = {
                                "invoice_id": row[0],
                                "customer": row[1],
                                "product": row[2],
                                "category": row[3],
                                "original_price": row[4],
                                "discounted_price": row[5],
                                "discount": row[6],
                                "product_link": row[7],
                                "context": row[8],
                                "sentiment_json": row[9],
                            }
                            st.markdown("**Invoice Data**")
                            st.json(inv_dict)
                        else:
                            st.info("No invoice data found for this ID.")
                elif src.startswith("review_"):
                    rev_id = src.replace("review_", "")
                    sql = """
                        SELECT source, customer_name, product_name, sentiment_label, sentiment_score, sentiment
                        FROM review WHERE id = %s;
                    """
                    with psycopg2.connect(SYNC_DB_URI) as conn, conn.cursor() as cur:
                        cur.execute(sql, (rev_id,))
                        row = cur.fetchone()
                        if row:
                            review_dict = {
                                "source": row[0],
                                "customer_name": row[1],
                                "product_name": row[2],
                                "sentiment_label": row[3],
                                "sentiment_score": row[4],
                                "sentiment": row[5]
                            }
                            st.markdown("**Review Data**")
                            st.json(review_dict)
                        else:
                            st.info("No review data found for this ID.")
                else:
                    # raw metadata
                    raw = fetch_row("metadata_langchain_table","source","metadata",src)
                    st.markdown("**Metadata Insight**")
                    st.json(raw)

                    # semantic description
                    enr = fetch_row("metadata_langchain_enriched","source","enriched_description",src)
                    st.markdown("**Semantic Description**")
                    st.text(enr)

                    # relationships
                    rel = fetch_row("metadata_langchain_relationships","source","relationships",src)
                    st.markdown("**Relationships**")
                    if isinstance(rel, (dict, list)):
                        st.json(rel)
                    else:
                        st.text(rel)

                st.markdown("---")

if "smart tv" in query.lower() and "invoice" in query.lower() and "negative" in query.lower():
    results = get_top_invoices_with_negative_reviews(product_keyword="tv", top_n=3)
    st.subheader("Top 3 Smart TV Invoices with Negative Reviews")
    for i, item in enumerate(results, 1):
        st.markdown(f"### {i}. Invoice ID: {item['invoice']['invoice_id']}")
        st.json(item['invoice'])
        st.markdown("**Negative Review:**")
        st.json(item['review'])
        st.markdown("---")
else:
    # fallback to normal RAG
    pass

# Debugging: show all available tables and columns
if st.button("Show All Tables & Columns"):
    with st.expander("Click to expand", expanded=True):
        try:
            with psycopg2.connect(SYNC_DB_URI) as conn:
                # Fetch all table names
                tables = fetch_all("information_schema.tables", ["table_name"])
                for table in tables:
                    table_name = table[0]
                    st.write(f"**Table:** {table_name}")

                    # Fetch and display all columns for this table
                    columns = fetch_all(
                        "information_schema.columns",
                        ["column_name"],
                        f"table_name = '{table_name}'"
                    )
                    col_names = [col[0] for col in columns]
                    st.write(f"  Columns: {', '.join(col_names)}")
        except Exception as e:
            st.error(f"Error fetching tables: {e}")
