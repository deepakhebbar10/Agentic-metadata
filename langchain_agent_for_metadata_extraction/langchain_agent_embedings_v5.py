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

    def extract_db_metadata(self):
        with psycopg2.connect(SYNC_DB_URI) as conn, conn.cursor() as cur:
            cur.execute("""
                SELECT column_name, data_type
                FROM information_schema.columns
                WHERE table_name = 'product_reviews';
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

    def extract_pdf_metadata(self):
        # load already‐seen keys
        with psycopg2.connect(SYNC_DB_URI) as conn, conn.cursor() as cur:
            cur.execute("SELECT source FROM metadata_langchain_table;")
            seen = {r[0] for r in cur.fetchall()}

        pdf_meta = []
        for key in list_s3_keys(self.bucket, self.prefix):
            if not key.lower().endswith(".pdf"):
                continue
            if key in seen:
                # skip already processed
                continue

            data   = load_s3_bytes(self.bucket, key)
            reader = fitz.open(stream=io.BytesIO(data), filetype="pdf")
            full_txt = "".join(reader[i].get_text("text") for i in range(len(reader)))

            ins = self.analyze(f"Analyze this PDF and provide metadata insights:\n\n{full_txt}")
            pdf_meta.append({"source": key, "insights": ins})

        self.metadata["PDF_Metadata"] = pdf_meta

    def extract_csv_metadata(self):
        # load already‐seen keys
        with psycopg2.connect(SYNC_DB_URI) as conn, conn.cursor() as cur:
            cur.execute("SELECT source FROM metadata_langchain_table;")
            seen = {r[0] for r in cur.fetchall()}

        csv_meta = []
        for key in list_s3_keys(self.bucket, self.prefix):
            if not key.lower().endswith(".csv"):
                continue
            if key in seen:
                # skip already processed
                continue

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
            csv_meta.append({"source": key, "insights": ins})

        self.metadata["CSV_Metadata"] = csv_meta

    def persist(self):
        # merge into JSON file
        on_disk = {}
        if os.path.exists("metadata.json"):
            on_disk = json.load(open("metadata.json"))
        on_disk.update(self.metadata)
        with open("metadata.json","w") as f:
            json.dump(on_disk, f, indent=4)

        # upsert into Postgres
        with psycopg2.connect(SYNC_DB_URI) as conn, conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS metadata_langchain_table(
                    id SERIAL PRIMARY KEY,
                    source TEXT UNIQUE,
                    metadata JSONB
                );
            """)
            for src, meta in [
                ("DB_Columns",  self.metadata["DB_Columns"]),
                ("DB_Insights", self.metadata["DB_Insights"]),
            ] + [(m["source"], m) for m in self.metadata.get("PDF_Metadata", [])] \
              + [(m["source"], m) for m in self.metadata.get("CSV_Metadata", [])]:
                cur.execute(
                    """
                    INSERT INTO metadata_langchain_table(source, metadata)
                    VALUES (%s, %s)
                    ON CONFLICT (source) DO UPDATE
                      SET metadata = EXCLUDED.metadata
                    """,
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
            return [{"source": r[0], "metadata": r[1]} for r in cur.fetchall()]

    def fetch_already_enriched(self):
        with psycopg2.connect(SYNC_DB_URI) as conn, conn.cursor() as cur:
            cur.execute("SELECT source FROM metadata_langchain_enriched;")
            return {r[0] for r in cur.fetchall()}

    def enrich_metadata(self):
        raw_entries = self.fetch_raw()
        seen_sources = self.fetch_already_enriched()

        new_items = []
        for entry in raw_entries:
            src = entry["source"]
            if src in seen_sources:
                continue  # skip already enriched

            prompt = (
                f"Given the following metadata field: {entry['metadata']}\n"
                f"- Generate a meaningful description.\n"
                f"- Identify its semantic category (e.g., \"Price\", \"Review Data\", \"Product Details\").\n"
                f"- Suggest any missing values.\n"
            )
            desc = self.analyze(prompt)
            new_items.append({
                "source": src,
                "original": entry["metadata"],
                "description": desc
            })

        # Merge into JSON on disk
        enriched_path = "metadata_langchain_enriched.json"
        existing = {}
        if os.path.exists(enriched_path):
            existing = json.load(open(enriched_path))
        all_enriched = existing.get("Enriched", []) + new_items
        self.metadata = {"Enriched": all_enriched}
        with open(enriched_path, "w") as f:
            json.dump(self.metadata, f, indent=4)

        # Upsert into Postgres
        with psycopg2.connect(SYNC_DB_URI) as conn, conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS metadata_langchain_enriched(
                    id SERIAL PRIMARY KEY,
                    source TEXT UNIQUE,
                    original_metadata JSONB,
                    enriched_description TEXT
                );
            """)
            for item in new_items:
                cur.execute(
                    """
                    INSERT INTO metadata_langchain_enriched
                      (source, original_metadata, enriched_description)
                    VALUES (%s, %s, %s)
                    ON CONFLICT (source) DO NOTHING;
                    """,
                    (item["source"], json.dumps(item["original"]), item["description"])
                )
            conn.commit()

        return self.metadata

# ── Relationship Discovery (unchanged) ─────────────────────────────────────────
class RelationshipDiscoveryAgent(BaseAgent):
    def __init__(self, use_gemini=False):
        super().__init__(use_gemini=use_gemini)
        with psycopg2.connect(SYNC_DB_URI) as conn, conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS metadata_langchain_relationships (
                    id SERIAL PRIMARY KEY,
                    source TEXT UNIQUE,
                    relationships JSONB
                );
            """)

    def fetch_enriched(self):
        with psycopg2.connect(SYNC_DB_URI) as conn, conn.cursor() as cur:
            cur.execute("SELECT source, enriched_description FROM metadata_langchain_enriched;")
            return cur.fetchall()

    def discover_relationships(self) -> dict:
        all_entries  = self.fetch_enriched()
        relationships = {}

        with psycopg2.connect(SYNC_DB_URI) as conn, conn.cursor() as cur:
            cur.execute("SELECT source FROM metadata_langchain_relationships;")
            done = {r[0] for r in cur.fetchall()}

        for src, desc in all_entries:
            if src in done:
                continue

            others = [
                f"- `{o_src}`: \"{o_desc}\""
                for o_src, o_desc in all_entries if o_src != src
            ]
            others_block = "\n".join(others)

            prompt = f"""
You are a metadata-analysis AI.

The target field is `{src}`, with the enriched description:

\"\"\"{desc}\"\"\"

Below are other fields (with their file or DB names):

{others_block}

Please:
1. Determine which of these fields are related to `{src}`.
2. For each related field, state the file or database where it appears.
3. In brief describe the nature of the relationship based on the content of the fields.
   (e.g., same product, similar price range, related product to same comapny, shared discount pattern).


Constraints:it should not give relationship because it belongs to same type of files like amazon invoices etc.
there should be actual relationship present like the relationship should not be like if files are following 
the same structure with the product name,original and discounted price , discount percentage then there should 
not be relationship the relationship should not be like if files  Shares the structure of containing 
key-value pair.The relationship should be based on the contents and not in the fields or types .

Example of how the relationship should be:
Scenario1:invoice112 file and invoice114 file have the same product  MI Usb Type-C Cable Smartphone.
description should be like Shares the product MI Usb Type-C Cable Smartphone.
   Both files identify the same product.
Scenario2:invoice112 file and invoice114 file have the product  related to USB
description should be like Shares similar product of USB is found between the two files

 
Example of how the relationship should not be:Shares the characteristic of analyzing Amazon purchase invoices. 
   Both identify key information like product name, prices, discounts, and customer details.

the above scenario are just examples and not the actual relationship. so you should form similar 
relationships the relationship need not be just on product it can be on the same customer name 
or same discount pattern etc.

note:if no relationships are found just mention no relationship found and do not 
compare missing values similarity when comparing

Return valid JSON where each key is a related field name, and its value 
is an object with:
- "file": the source file or database 
- "relationship":  description 
"""

            raw = self.analyze(prompt)
            try:
                rel_map = json.loads(raw)
            except json.JSONDecodeError:
                rel_map = {"_text": raw}

            with psycopg2.connect(SYNC_DB_URI) as conn, conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO metadata_langchain_relationships(source,relationships) "
                    "VALUES (%s,%s) ON CONFLICT DO NOTHING;",
                    (src, json.dumps(rel_map))
                )
                conn.commit()

            relationships[src] = rel_map

        # --- merge into metadata_langchain_relationships.json ---
        existing = {}
        if os.path.exists("metadata_langchain_relationships.json"):
            with open("metadata_langchain_relationships.json") as f:
                existing = json.load(f)
        old = {k: v for k, v in existing.items()}
        old.update(relationships)
        with open("metadata_langchain_relationships.json", "w") as f:
            json.dump(old, f, indent=4)

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
