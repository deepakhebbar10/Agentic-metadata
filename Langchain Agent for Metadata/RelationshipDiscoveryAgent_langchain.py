#!/usr/bin/env python3
import os
import ssl
import json
import re
import psycopg2
import ollama
import google.generativeai as genai
from transformers import pipeline
from langchain_huggingface import HuggingFacePipeline

# ── TLS / protobuf hack for Windows ───────────────────────────────────────────────
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
ssl._create_default_https_context = ssl._create_unverified_context

# ── DB configuration ─────────────────────────────────────────────────────────────
DB_CONFIG = {
    "dbname":   "langchain_metadata",
    "user":     "postgres",
    "password": "deepak",
    "host":     "localhost",
    "port":     "5432",
}
SYNC_DB_URI = (
    f"postgresql://{DB_CONFIG['user']}:{DB_CONFIG['password']}@"
    f"{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['dbname']}"
)

# ── LLM configuration ────────────────────────────────────────────────────────────
HF_MODEL = os.getenv("HF_MODEL", "gpt2")
GEMINI_MODEL = "gemini-2.0-flash"
GEMINI_KEY_FILE = "gemini_api_key.txt"

def load_gemini_api_key(filename: str = GEMINI_KEY_FILE) -> str:
    path = os.path.join(os.path.dirname(__file__), filename)
    with open(path, "r") as f:
        return f.read().strip()

def generate_with_gemini(prompt: str) -> str:
    genai.configure(api_key=load_gemini_api_key())
    model = genai.GenerativeModel(GEMINI_MODEL)
    resp  = model.generate_content(prompt)
    return getattr(resp, "text", str(resp))

# ── Base agent with Gemini, HF, and Ollama support ───────────────────────────────
class BaseAgent:
    def __init__(self, *, use_gemini: bool = False, use_hf: bool = True):
        self.use_gemini = use_gemini
        self.use_hf     = use_hf
        self.llm        = None
        if self.use_hf:
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
                print(f"[WARN] Gemini call failed: {e}")

        # 2) Try HuggingFace
        if self.use_hf and self.llm is not None:
            return self.llm(prompt).strip()

        # 3) Fallback to Ollama/Mistral
        resp = ollama.chat("mistral", messages=[{"role": "user", "content": prompt}])
        if isinstance(resp, dict) and "message" in resp:
            return resp["message"].strip()
        if hasattr(resp, "message"):
            return resp.message.strip()
        return str(resp).strip()

# ── Infer table names from filename substrings ───────────────────────────────────
FILE_TYPE_TABLE_MAP = {
    "invoice": "invoice",
    "review":  "review",
    "sales":   "sales_summary",
}

def infer_table_name(source: str) -> str:
    lower = source.lower()
    for key, tbl in FILE_TYPE_TABLE_MAP.items():
        if key in lower:
            return tbl
    return "generic_documents"

# ── Agent to discover schema-level relationships ─────────────────────────────────
class SchemaRelationshipAgent(BaseAgent):
    def __init__(self, *, use_gemini: bool = False, use_hf: bool = True):
        super().__init__(use_gemini=use_gemini, use_hf=use_hf)

    def fetch_table_schemas(self) -> dict:
        """
        Return a dict mapping table_name → list of (column_name, data_type),
        excluding tables starting with 'metadata_langchain'.
        """
        schemas = {}
        with psycopg2.connect(SYNC_DB_URI) as conn, conn.cursor() as cur:
            cur.execute("""
                SELECT table_name
                FROM information_schema.tables
                WHERE table_schema = 'public'
                  AND table_type = 'BASE TABLE'
                  AND table_name NOT LIKE 'metadata_langchain%%'
                  AND table_name NOT LIKE 'relationships';
            """)
            tables = [r[0] for r in cur.fetchall()]
            for tbl in tables:
                cur.execute("""
                    SELECT column_name, data_type
                    FROM information_schema.columns
                    WHERE table_schema = 'public'
                      AND table_name = %s
                    ORDER BY ordinal_position;
                """, (tbl,))
                schemas[tbl] = cur.fetchall()
        return schemas

    def discover_relationships(self) -> list:
        schemas = self.fetch_table_schemas()
        # Build a schema description block
        lines = []
        for tbl, cols in schemas.items():
            cols_str = ", ".join(f"{c} ({t})" for c, t in cols)
            lines.append(f"Table `{tbl}`: {cols_str}")
        schema_block = "\n".join(lines)

        prompt = f"""
You are a metadata-analysis AI. Below are the schemas of several tables:

{schema_block}

Identify all meaningful relationships between these tables,
Include:
  • Explicit foreign-key style links (matching column names).
  • Explicit key style links (matching column contents).
  • ternary relationships between tables.
  • Semantic links (e.g. customer_id ↔ user_id, shared business concepts).

Return a JSON array of objects with:
  - "table1": first table name
  - "table2": second table name
  - "relationship": brief description
  - "relationship_type": e.g., One-to-Many, Many-to-Many, ForeignKey, PrimaryKey, Ternary
  - "description": brief description of the relationship
If no relationships found, return an empty array.
"""

        raw = self.analyze(prompt)
        # Remove code block markers if present
        cleaned = re.sub(r"```(?:json)?\s*|\s*```", "", raw, flags=re.IGNORECASE).strip()
        try:
            rels = json.loads(cleaned)
            if not isinstance(rels, list):
                raise ValueError("Expected a JSON array")
        except Exception:
            rels = [{"_error": cleaned}]

        return rels

if __name__ == "__main__":
    # 1) discover your table–table relationships (as before)
    agent = SchemaRelationshipAgent(use_gemini=True, use_hf=False)
    table_rels = agent.discover_relationships()
    print("Table→Table relationships:", json.dumps(table_rels, indent=2))

    # --- Store table-level relationships ---
    with psycopg2.connect(SYNC_DB_URI) as conn, conn.cursor() as cur:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS table_relationship (
                table1            TEXT,
                table2            TEXT,
                relationship      TEXT,
                relationship_type TEXT,
                description       TEXT
            );
        """)
        for rel in table_rels:
            if "_error" in rel:
                print("Skipping error result:", rel["_error"])
                continue
            cur.execute(
                "INSERT INTO table_relationship (table1, table2, relationship, relationship_type, description) VALUES (%s, %s, %s, %s, %s);",
                (
                    rel.get("table1"),
                    rel.get("table2"),
                    rel.get("relationship"),
                    rel.get("relationship_type"),
                    rel.get("description")
                )
            )
        conn.commit()

    # 2) build a schema block for prompting
    schemas = agent.fetch_table_schemas()
    schema_block = "\n".join(
        f"• `{tbl}`: " + ", ".join(f"{col} ({dtype})" for col, dtype in cols)
        for tbl, cols in schemas.items()
    )

    # 3) load your file‐level enriched metadata
    files = []
    with psycopg2.connect(SYNC_DB_URI) as conn, conn.cursor() as cur:
        cur.execute("SELECT source, original_metadata FROM metadata_langchain_enriched;")
        for src, desc in cur.fetchall():
            files.append({"source": src, "desc": desc})

    # 4) for each file, ask the LLM which tables it relates to
    file_table_links = []
    for f in files:
        prompt = f"""
You are Metadata Relationship Agent.

Here are your SQL table schemas:
{schema_block}

Here is the enriched metadata for file `{f['source']}`:
\"\"\"{f['desc']}\"\"\"

Determine which of these tables this file is related to, and why.
Return a JSON array of objects, each with:
  - "file": the file name
  - "table": the table name
  - "relationship_type":e.g., One-to-Many, Many-to-Many, ForeignKey, PrimaryKey, Ternary etc.
  - "description": a brief explanation

If none, return an empty array.
"""
        raw = agent.analyze(prompt)
        cleaned = re.sub(r"```(?:json)?\s*|\s*```", "", raw, flags=re.IGNORECASE).strip()
        if not cleaned:
                print(f"[WARN] No JSON returned for {f['source']}, skipping.")
                continue

        try:
            matches = json.loads(cleaned)
            if not isinstance(matches, list):
                raise ValueError("Expected list")
        except Exception as e:
            print(f"[WARN] could not parse file→table for {f['source']}: {e}\nCleaned raw:\n{cleaned}\n")
            continue

        for m in matches:
            file_table_links.append(m)

    # 5) Persist into a SQL table named Relationships
    with psycopg2.connect(SYNC_DB_URI) as conn, conn.cursor() as cur:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS Relationships (
              file1             TEXT,
              file2             TEXT,
              relationship_type TEXT,
              description       TEXT
            );
        """)
        for link in file_table_links:
            cur.execute(
                "INSERT INTO Relationships (file1, file2, relationship_type, description) VALUES (%s,%s,%s,%s);",
                (
                    link.get("file"),
                    link.get("table"),
                    link.get("relationship_type"),
                    link.get("description")
                )
            )
        conn.commit()

    print("File→Table relationships saved:")
    for r in file_table_links:
        print(f"  {r['file']} → {r['table']}  [{r['relationship_type']}] {r['description']}")