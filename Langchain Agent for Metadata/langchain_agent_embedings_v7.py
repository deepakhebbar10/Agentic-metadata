#!/usr/bin/env python3
import os
import io
import ssl
import json
import re
import time
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
    paginator = _s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            yield obj["Key"]

def load_s3_bytes(bucket: str, key: str) -> bytes:
    resp = _s3.get_object(Bucket=bucket, Key=key)
    return resp["Body"].read()

# ── Configuration ───────────────────────────────────────────────────────────────
DB_CONFIG = {
    "dbname":   "langchain_metadata",
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

# ── Helpers for dynamic table storage ────────────────────────────────────────────
FILE_TYPE_TABLE_MAP = {
    "invoice":       "invoice",
    "review":        "review",
    "sales":         "sales_summary",
    # add more mappings here as needed
}

def infer_table_name(source: str) -> str:
    lower = source.lower()
    for key, table in FILE_TYPE_TABLE_MAP.items():
        if key in lower:
            return table
    return "generic_documents"

def normalize_col(col: str) -> str:
    col = col.strip().lower()
    col = re.sub(r"\W+", "_", col)
    col = re.sub(r"_+", "_", col)
    return col.strip("_") or "field"

def store_insights(insights: dict, db_url: str):
    # 1) Normalize incoming entities into a list of (key, value) pairs
    raw_entities = insights.get("entities", {})
    kvs = []
    if isinstance(raw_entities, dict):
        kvs = list(raw_entities.items())
    elif isinstance(raw_entities, list):
        for ent in raw_entities:
            if isinstance(ent, dict) and len(ent) == 1:
                kvs.append(next(iter(ent.items())))

    # 2) Decide table
    table = infer_table_name(insights["source"])

    # 3) REVIEWS → one row per sentiment entry
    if table == "review":
        with psycopg2.connect(db_url) as conn, conn.cursor() as cur:
            cur.execute("""
              CREATE TABLE IF NOT EXISTS review (
                id SERIAL PRIMARY KEY,
                source          TEXT,
                customer_name   TEXT,
                product_name    TEXT,
                sentiment_label TEXT,
                sentiment_score REAL,
                sentiment       TEXT
              );
            """)
            for entry in insights.get("sentiment", []):
                cur.execute("""
                  INSERT INTO review
                    (source, customer_name, product_name,
                     sentiment_label, sentiment_score, sentiment)
                  VALUES (%s, %s, %s, %s, %s, %s);
                """, (
                  insights["source"],
                  entry.get("customer_name"),
                  entry.get("product_name"),
                  entry.get("sentiment_label"),
                  entry.get("sentiment_score"),
                  entry.get("sentiment"),
                ))
            conn.commit()
        return

    # 4) SALES SUMMARY → one row per category entry
    if table == "sales_summary":
        with psycopg2.connect(db_url) as conn, conn.cursor() as cur:
            cur.execute("""
              CREATE TABLE IF NOT EXISTS sales_summary (
                id                SERIAL PRIMARY KEY,
                source            TEXT,
                category          TEXT,
                total_sales_amount TEXT,
                average_price     TEXT,
                product_count     TEXT
              );
            """)
            # assume each ent dict holds one summary row
            for ent in insights.get("entities", []):
                data = { normalize_col(k): v for k, v in ent.items() }
                # insert with default None if missing
                cur.execute("""
                  INSERT INTO sales_summary
                    (source, category, total_sales_amount, average_price, product_count)
                  VALUES (%s, %s, %s, %s, %s);
                """, (
                  insights["source"],
                  data.get("category"),
                  data.get("total_sales_amount"),
                  data.get("average_price"),
                  data.get("product_count"),
                ))
            conn.commit()
        return

    # 5) INVOICE → add keywords column and all required fields
    if table == "invoice":
        # Extract fields from entities if not present at top-level
        entity_map = {}
        raw_entities = insights.get("entities", {})
        if isinstance(raw_entities, dict):
            entity_map = {k.lower(): v for k, v in raw_entities.items()}
        elif isinstance(raw_entities, list):
            for ent in raw_entities:
                if isinstance(ent, dict):
                    for k, v in ent.items():
                        entity_map[k.lower()] = v

        def get_field(name):
            # Normalize: lowercase, remove spaces and underscores
            def norm(s):
                return s.replace(" ", "").replace("_", "").lower()
            # Try top-level
            if name in insights and insights[name] is not None:
                return insights[name]
            # Try normalized top-level
            for k, v in insights.items():
                if norm(k) == norm(name) and v is not None:
                    return v
            # Try entities
            for k, v in entity_map.items():
                if norm(k) == norm(name) and v is not None:
                    return v
            return None

        with psycopg2.connect(db_url) as conn, conn.cursor() as cur:
            cur.execute("""
              CREATE TABLE IF NOT EXISTS invoice (
                id SERIAL PRIMARY KEY,
                invoice_id TEXT,
                customer TEXT,
                product TEXT,
                category TEXT,
                original_price TEXT,
                discounted_price TEXT,
                discount TEXT,
                product_link TEXT,
                context TEXT,
                sentiment_json TEXT,
                keywords TEXT,
                source TEXT
              );
            """)
            cur.execute("""
              INSERT INTO invoice (
                invoice_id, customer, product, category, original_price,
                discounted_price, discount, product_link, context,
                sentiment_json, keywords, source
              ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s);
            """, (
              get_field("invoice_id"),
              get_field("customer"),
              get_field("product"),
              get_field("category"),
              get_field("original_price"),
              get_field("discounted_price"),
              get_field("discount"),
              get_field("product_link"),
              insights.get("context"),
              json.dumps(insights.get("sentiment")) if insights.get("sentiment") is not None else None,
              insights.get("Keywords"),
              insights.get("source")
            ))
            conn.commit()
        return

    # 5) FALLBACK for invoices, generic docs, etc.
    #    Build a deduped list of columns+values
    cols, vals = [], []
    for key, value in kvs:
        col = normalize_col(key)
        if col in cols:
            continue
        cols.append(col)
        vals.append(value)

    # append context & sentiment_json once
    if "context" in insights and "context" not in cols:
        cols.append("context")
        vals.append(insights["context"])
    if "sentiment" in insights and "sentiment_json" not in cols:
        cols.append("sentiment_json")
        vals.append(json.dumps(insights["sentiment"]))

    # DDL + DML
    col_defs   = ",\n    ".join(f"{c} TEXT" for c in cols)
    create_sql = f"""
    CREATE TABLE IF NOT EXISTS {table} (
      id SERIAL PRIMARY KEY,
      {col_defs}
    );
    """
    placeholders = ", ".join("%s" for _ in cols)
    insert_sql   = f"""
      INSERT INTO {table} ({', '.join(cols)}) 
      VALUES ({placeholders});
    """

    with psycopg2.connect(db_url) as conn, conn.cursor() as cur:
        cur.execute(create_sql)
        cur.execute(insert_sql, vals)
        conn.commit()


# ── Base agent (no local embeddings) ────────────────────────────────────────────
class BaseAgent:
    def __init__(self, *, use_gemini: bool = False, use_hf: bool = True):
        self.use_gemini = use_gemini
        self.use_hf     = use_hf
        self.llm        = None
        if self.use_hf and not self.use_gemini:
            hf_pipe = pipeline(
                "text-generation",
                model=HF_MODEL,
                framework="pt",
                max_length=512,
                truncation=True
            )
            self.llm = HuggingFacePipeline(pipeline=hf_pipe)

    def analyze(self, prompt: str) -> str:
        # 1) Try Gemini
        if self.use_gemini:
            try:
                return generate_with_gemini(prompt).strip()
            except Exception as e:
                print(f"[WARN] Gemini failed: {e}, falling back…")

        # 2) Try local HF
        if self.use_hf and self.llm is not None:
            return self.llm(prompt).strip()

        # 3) Fallback to Ollama
        resp = ollama.chat("mistral", messages=[{"role":"user","content":prompt}])
        if isinstance(resp, dict) and "message" in resp:
            return resp["message"].strip()
        if hasattr(resp, "message"):
            return resp.message.strip()
        return str(resp).strip()

# ── Metadata Discovery + Enrichment ─────────────────────────────────────────────
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
    f"Analyze DB metadata for table `product_reviews`:\n"
    f"Columns={self.metadata['DB_Columns']}\n"
    f"Row count={nrows}\n"
    f"Sample={self.metadata['DB_Sample_Data']}\n\n"
    "Return a JSON object with:\n"
    "  • entities: list of named entities (e.g., customer names, product names,price etc)\n"
    "  • relationships: any relationships between those entities like primary key foreign key\n"
    "  • context: a brief summary of what this table represents\n"
    "  • sentiment: for each sample review, an array of {row_index, sentiment_label, sentiment_score}\n"
)
        self.metadata["DB_Insights"] = self.analyze(insight_p)
		
		
    def extract_pdf_metadata(self):
     with psycopg2.connect(SYNC_DB_URI) as conn, conn.cursor() as cur:
        cur.execute("SELECT source FROM metadata_langchain_table;")
        seen = {r[0] for r in cur.fetchall()}

     for key in list_s3_keys(self.bucket, self.prefix):
        if not key.lower().endswith(".pdf") or key in seen:
            continue

        data   = load_s3_bytes(self.bucket, key)
        reader = fitz.open(stream=io.BytesIO(data), filetype="pdf")
        full_txt = "".join(reader[i].get_text("text") for i in range(len(reader)))

        prompt = (
            "Analyze this PDF and return *only* a JSON object with:\n"
            "  • entities: list of simple KEY:VALUE strings (e.g. NAME:value, price:value…)\n"
            "  • context: summary of the document’s purpose\n"
            "  • sentiment: if reviews exist, array of {customer_name, product_name, sentiment_label, sentiment_score, sentiment}\n\n"
            "  • Keywords: Consists of all the Keywords found in the Pdf which can be of importance}\n"
            f"{full_txt}"
        )

        # 1) time & call
        t0 = time.time()
        raw = self.analyze(prompt)
        print(f"[DEBUG] LLM output for {key} ({time.time()-t0:.1f}s):\n{raw}\n")

        # 2) strip markdown fences
        cleaned = re.sub(r"```(?:json)?\s*|\s*```", "", raw, flags=re.IGNORECASE).strip()
        if not cleaned:
            print(f"[WARN] No JSON returned for {key}, skipping.")
            continue

        # 3) parse
        try:
            insights = json.loads(cleaned)
        except json.JSONDecodeError as err:
            print(f"[ERROR] JSON parse failed for {key}: {err}")
            print(">>> Cleaned payload:\n", cleaned)
            continue

        # 4) store & track
        insights["source"] = key
        store_insights(insights, SYNC_DB_URI)
        self.metadata.setdefault("PDF_Metadata", []).append(insights)

    def extract_csv_metadata(self):
        with psycopg2.connect(SYNC_DB_URI) as conn, conn.cursor() as cur:
            cur.execute("SELECT source FROM metadata_langchain_table;")
            seen = {r[0] for r in cur.fetchall()}

        for key in list_s3_keys(self.bucket, self.prefix):
            if not key.lower().endswith(".csv") or key in seen:
                continue

            # load & truncate CSV text
            data = load_s3_bytes(self.bucket, key)
            df   = pd.read_csv(io.BytesIO(data))
            csv_text = df.to_csv(index=False)
            if len(csv_text) > 50000:
                csv_text = csv_text[:50000] + "\n…(truncated)…"

            prompt = (
                "Analyze the following CSV data and return *only* a JSON object with:\n"
                "  • entities: list of simple KEY:VALUE strings (one per entry)\n"
                "  • context: a brief summary of the document’s purpose\n"
                "  • sentiment: if reviews present, array of {customer_name, product_name, sentiment_label, sentiment_score, sentiment}\n\n"
                "  • Keywords: Consists of all the Keywords found in the Pdf which can be of importance}\n"
                f"{csv_text}"
            )

            # 1) call & time
            t0 = time.time()
            raw = self.analyze(prompt)
            print(f"[DEBUG] LLM output for {key} ({time.time() - t0:.1f}s):\n{raw}\n")

            # 2) strip any ```json fences
            cleaned = re.sub(r"```(?:json)?\s*|\s*```", "", raw, flags=re.IGNORECASE).strip()
            if not cleaned:
                print(f"[WARN] No JSON returned for {key}, skipping.")
                continue

            # 3) parse
            try:
                insights = json.loads(cleaned)
            except json.JSONDecodeError as err:
                print(f"[ERROR] JSON parse failed for {key}: {err}")
                print(">>> Cleaned payload:\n", cleaned)
                continue

            # 4) store
            insights["source"] = key
            store_insights(insights, SYNC_DB_URI)
            self.metadata.setdefault("CSV_Metadata", []).append(insights)

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
     raw_entries   = self.fetch_raw()
     seen_sources  = self.fetch_already_enriched()
     new_items     = []

     for entry in raw_entries:
         src = entry["source"]
         if src in seen_sources:
             continue

         prompt = (
            f"Here is some raw metadata for `{src}`:\n"
            f"{json.dumps(entry['metadata'], indent=2)}\n\n"
            "Please return valid JSON with exactly these keys:\n"
                f"- Generate a meaningful description.\n"
                f"- Identify its semantic category (e.g., \"Price\", \"Review Data\", \"Product Details\").\n"
                f"- Suggest any missing values.\n"
        )
         enriched_json = self.analyze(prompt).strip()
        # you may want to json.loads() this to validate

         new_items.append({
            "source": src,
            "original": entry["metadata"],
            "description": enriched_json
        })


     with psycopg2.connect(SYNC_DB_URI) as conn, conn.cursor() as cur:
        cur.execute("""
          CREATE TABLE IF NOT EXISTS metadata_langchain_enriched (
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
              ON CONFLICT (source) DO UPDATE
                SET
                  original_metadata    = EXCLUDED.original_metadata,
                  enriched_description = EXCLUDED.enriched_description
              """,
              (item["source"], json.dumps(item["original"]), item["description"])
            )
        conn.commit()

     return {"Enriched": new_items}


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
    # 1️⃣ Discover raw metadata
    discovery = MetadataDiscoveryAgent(
        use_gemini=True,
        use_hf=False,
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
