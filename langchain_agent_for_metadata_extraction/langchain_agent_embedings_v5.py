#!/usr/bin/env python3
import os
import io
import ssl
import json
import argparse
import boto3
import fitz               # PyMuPDF for PDFs
import pandas as pd
import psycopg2
import google.generativeai as genai
import ollama             # Mistral via Ollama
from langchain_community.llms import HuggingFacePipeline
from langchain.schema import Document

# ── TLS / protobuf hack (Windows) ───────────────────────────────────────────────
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
ssl._create_default_https_context = ssl._create_unverified_context

# ── Argparse for S3 details ──────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Metadata pipeline (DB + S3)")
parser.add_argument("--bucket",  required=True, help="S3 bucket name")
parser.add_argument("--prefix",  default="",    help="S3 key prefix (folder/)")
args = parser.parse_args()

# ── Load AWS creds from CSV & set env vars ──────────────────────────────────────
_creds = pd.read_csv("Deepak_accessKeys.csv")
os.environ["AWS_ACCESS_KEY_ID"]     = _creds.loc[0, "Access key ID"]
os.environ["AWS_SECRET_ACCESS_KEY"] = _creds.loc[0, "Secret access key"]

# ── Boto3 client ────────────────────────────────────────────────────────────────
_s3 = boto3.client("s3")

def list_s3_keys(bucket: str, prefix: str):
    """Yield all object keys under this bucket/prefix."""
    paginator = _s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            yield obj["Key"]

def load_s3_bytes(bucket: str, key: str) -> bytes:
    """Download an object and return its bytes."""
    resp = _s3.get_object(Bucket=bucket, Key=key)
    return resp["Body"].read()

# ── Configuration ───────────────────────────────────────────────────────────────
DB_CONFIG = {
    "dbname":   "metadata",
    "user":     "postgres",
    "password": "deepak",
    "host":     "localhost",
    "port":     "5432",
}
DATA_BUCKET = args.bucket
DATA_PREFIX = args.prefix

SYNC_DB_URI = (
    f"postgresql://{DB_CONFIG['user']}:{DB_CONFIG['password']}@"
    f"{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['dbname']}"
)

GEMINI_API_KEY_FILE = "gemini_api_key.txt"
GEMINI_MODEL        = "gemini-2.0-flash"
HF_MODEL            = os.getenv("HF_MODEL", "gpt2")

# ── Gemini helper ───────────────────────────────────────────────────────────────
def load_gemini_api_key(filename: str) -> str:
    path = os.path.join(os.path.dirname(__file__), filename)
    with open(path, "r") as f:
        return f.read().strip()

def generate_with_gemini(prompt: str) -> str:
    genai.configure(api_key=load_gemini_api_key(GEMINI_API_KEY_FILE))
    model = genai.GenerativeModel(GEMINI_MODEL)
    resp  = model.generate_content(prompt)
    return resp.text if hasattr(resp, "text") else str(resp)

# ── Base agent (no local embeddings) ────────────────────────────────────────────
class BaseAgent:
    def __init__(self, *, use_gemini: bool = False, use_hf: bool = True):
        self.use_gemini = use_gemini
        self.llm = None
        if use_hf and not use_gemini:
            from transformers import pipeline
            hf_pipe = pipeline(
                "text-generation",
                model=HF_MODEL,
                framework="pt",
                max_length=512,
                truncation=True
            )
            self.llm = HuggingFacePipeline(pipeline=hf_pipe)

    def analyze(self, prompt: str) -> str:
        if self.use_gemini:
            try:
                return generate_with_gemini(prompt)
            except Exception as e:
                print(f"[WARN] Gemini failed, falling back to Ollama: {e}")
        # fallback → Ollama/Mistral
        resp = ollama.chat("mistral", messages=[{"role":"user","content":prompt}])
        return resp.get("message", str(resp))

# ── Metadata Discovery (now reading from S3) ──────────────────────────────────
class MetadataDiscoveryAgent(BaseAgent):
    def __init__(self, *, use_gemini=False, use_hf=True,
                 bucket: str = None, prefix: str = ""):
        super().__init__(use_gemini=use_gemini, use_hf=use_hf)
        self.bucket = bucket
        self.prefix = prefix
        self.metadata = {}

    # --- unchanged DB extraction ---
    def extract_db_metadata(self):
        with psycopg2.connect(SYNC_DB_URI) as conn, conn.cursor() as cur:
            cur.execute("""
                SELECT column_name,data_type
                FROM information_schema.columns
                WHERE table_name='product_reviews';
            """)
            cols = cur.fetchall()
            cur.execute("SELECT COUNT(*) FROM product_reviews;")
            nrows = cur.fetchone()[0]
            cur.execute("SELECT * FROM product_reviews LIMIT 5;")
            sample = cur.fetchall()
        self.metadata.update({
            "DB_Columns":     [{"name":c[0],"type":c[1]} for c in cols],
            "DB_Row_Count":   nrows,
            "DB_Sample_Data": [list(r) for r in sample],
        })
        insight_p = (
            f"Analyze DB metadata:\n"
            f"Columns={self.metadata['DB_Columns']}\n"
            f"Row count={nrows}\n"
            f"Sample={self.metadata['DB_Sample_Data']}"
        )
        self.metadata["DB_Insights"] = self.analyze(insight_p)

    # --- PDFs from S3 ---
    def extract_pdf_metadata(self):
        pdf_meta = []
        for key in list_s3_keys(self.bucket, self.prefix):
            if key.lower().endswith(".pdf"):
                data     = load_s3_bytes(self.bucket, key)
                reader   = fitz.open(stream=io.BytesIO(data), filetype="pdf")
                full_txt = "".join(reader[i].get_text("text") for i in range(len(reader)))
                ins = self.analyze(f"Analyze this PDF and provide metadata insights:\n\n{full_txt}")
                pdf_meta.append({"s3_key": key, "insights": ins})
        self.metadata["PDF_Metadata"] = pdf_meta

    # --- CSVs from S3 ---
    def extract_csv_metadata(self):
        csv_meta = []
        for key in list_s3_keys(self.bucket, self.prefix):
            if key.lower().endswith(".csv"):
                data = load_s3_bytes(self.bucket, key)
                df   = pd.read_csv(io.BytesIO(data))
                cols   = df.columns.tolist()
                dtypes = df.dtypes.apply(lambda x: x.name).to_dict()
                count  = len(df)
                sample = df.head(5).to_dict("records")
                ins = self.analyze(
                    f"Analyze CSV metadata:\n"
                    f"Columns={cols}\n"
                    f"Types={dtypes}\n"
                    f"Rows={count}\n"
                    f"Sample={sample}"
                )
                csv_meta.append({"s3_key": key, "insights": ins})
        self.metadata["CSV_Metadata"] = csv_meta

    # --- JSON + DB persistence unchanged ---
    def persist(self):
        with open("metadata.json","w") as f:
            json.dump(self.metadata, f, indent=4)
        with psycopg2.connect(SYNC_DB_URI) as conn, conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS metadata_langchain_table(
                    id SERIAL PRIMARY KEY, source TEXT, metadata JSONB
                );
            """)
            entries = [
                ("DB_Columns",  self.metadata["DB_Columns"]),
                ("DB_Insights", self.metadata["DB_Insights"]),
            ]
            for m in self.metadata.get("PDF_Metadata", []):
                entries.append((m["s3_key"], m))
            for m in self.metadata.get("CSV_Metadata", []):
                entries.append((m["s3_key"], m))
            for src, meta in entries:
                cur.execute(
                    "INSERT INTO metadata_langchain_table(source,metadata) VALUES (%s,%s) "
                    "ON CONFLICT DO NOTHING;",
                    (src, json.dumps(meta))
                )
            conn.commit()

    def extract_metadata(self):
        self.extract_db_metadata()
        self.extract_pdf_metadata()
        self.extract_csv_metadata()
        self.persist()
        return self.metadata

# ── Metadata Enrichment (unchanged) ────────────────────────────────────────────
class MetadataEnrichmentAgent(BaseAgent):
    def fetch_raw(self):
        with psycopg2.connect(SYNC_DB_URI) as conn, conn.cursor() as cur:
            cur.execute("SELECT source, metadata FROM metadata_langchain_table;")
            return [{"source":r[0], "metadata":r[1]} for r in cur.fetchall()]

    def enrich_metadata(self):
        enriched = []
        for entry in self.fetch_raw():
            prompt = (
                f"Given the following metadata field: {entry['metadata']}\n"
                f"- Generate a meaningful description.\n"
                f"- Identify its semantic category (e.g., \"Price\", \"Review Data\").\n"
                f"- Suggest any missing values.\n"
            )
            desc = self.analyze(prompt)
            enriched.append({
                "source": entry["source"],
                "original": entry["metadata"],
                "description": desc
            })
        self.metadata = {"Enriched": enriched}
        with open("metadata_langchain_enriched.json","w") as f:
            json.dump(self.metadata, f, indent=4)
        with psycopg2.connect(SYNC_DB_URI) as conn, conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS metadata_langchain_enriched(
                    id SERIAL PRIMARY KEY,
                    source TEXT,
                    original_metadata JSONB,
                    enriched_description TEXT
                );
            """)
            for item in enriched:
                cur.execute(
                    "INSERT INTO metadata_langchain_enriched(source,original_metadata,enriched_description) "
                    "VALUES (%s,%s,%s) ON CONFLICT DO NOTHING;",
                    (item["source"], json.dumps(item["original"]), item["description"])
                )
            conn.commit()
        return self.metadata

# ── Relationship Discovery (unchanged) ─────────────────────────────────────────
class RelationshipDiscoveryAgent(BaseAgent):
    def fetch_enriched(self):
        with psycopg2.connect(SYNC_DB_URI) as conn, conn.cursor() as cur:
            cur.execute("SELECT source, enriched_description FROM metadata_langchain_enriched;")
            return cur.fetchall()

    def discover_relationships(self):
        with psycopg2.connect(SYNC_DB_URI) as conn, conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS metadata_langchain_relationships(
                    id SERIAL PRIMARY KEY,
                    source TEXT UNIQUE,
                    relationships JSONB
                );
            """)
            conn.commit()

        relationships = {}
        for src, desc in self.fetch_enriched():
            prompt = (
                f"You are an AI assistant analyzing metadata relationships.\n"
                f"Given the metadata field `{src}` with enriched description:\n\n"
                f"\"{desc}\"\n\n"
                f"Explain its relationship with other metadata fields."
            )
            raw = self.analyze(prompt)
            try:
                rel_map = json.loads(raw)
            except:
                rel_map = {"text": raw}
            relationships[src] = rel_map

            with psycopg2.connect(SYNC_DB_URI) as conn, conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO metadata_langchain_relationships(source,relationships) "
                    "VALUES (%s,%s) ON CONFLICT(source) DO NOTHING;",
                    (src, json.dumps(rel_map))
                )
                conn.commit()

        with open("metadata_langchain_relationships.json","w") as f:
            json.dump(relationships, f, indent=4)
        return relationships

# ── Main orchestration ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    # 1️⃣ Discover raw metadata (DB + S3 PDFs/CSVs)
    discovery  = MetadataDiscoveryAgent(
        use_gemini=True,
        bucket=DATA_BUCKET,
        prefix=DATA_PREFIX
    )
    raw_meta   = discovery.extract_metadata()

    # 2️⃣ Enrich metadata
    enrichment = MetadataEnrichmentAgent(use_gemini=True)
    enriched   = enrichment.enrich_metadata()

    # 3️⃣ Discover relationships
    relation   = RelationshipDiscoveryAgent(use_gemini=True)
    relations  = relation.discover_relationships()

    # 📜 Print summary
    print(json.dumps({
        "metadata":      raw_meta,
        "enriched":      enriched,
        "relationships": relations
    }, indent=2))
    print("[INFO] Completed pipeline.")
