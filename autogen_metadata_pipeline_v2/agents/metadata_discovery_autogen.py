"""AutoGen AssistantAgent for discovering metadata from Postgres + PDFs."""
from typing import Dict, List
import os, json, fitz, psycopg2
from autogen_agentchat.agents import AssistantAgent
from autogen_core.tools import FunctionTool
from autogen_ext.models.openai import OpenAIChatCompletionClient

from config import DB_CONFIG, PDF_DIR

# ────────────  TOOL FUNCTIONS ────────────
def extract_db_metadata(db_cfg: Dict) -> Dict:
    """Return column list, row count, and 5‑row sample from product_reviews."""
    conn = psycopg2.connect(**db_cfg)
    cur = conn.cursor()
    cur.execute("""SELECT column_name, data_type
                    FROM information_schema.columns
                    WHERE table_name = 'product_reviews';""")
    columns = [{"name": c, "type": t} for c, t in cur.fetchall()]
    cur.execute("SELECT COUNT(*) FROM product_reviews;")
    rows = cur.fetchone()[0]
    cur.execute("SELECT * FROM product_reviews LIMIT 5;")
    col_names = [d[0] for d in cur.description]
    sample = [dict(zip(col_names, r)) for r in cur.fetchall()]
    conn.close()
    return {"columns": columns, "rows": rows, "sample": sample}

def extract_pdf_metadata(pdf_dir: str) -> List[Dict]:
    """Scan first three pages of each PDF and collect basic metadata."""
    out = []
    for fn in os.listdir(pdf_dir):
        if not fn.lower().endswith(".pdf"):
            continue
        path = os.path.join(pdf_dir, fn)
        doc = fitz.open(path)
        text = "".join(doc[p].get_text("text") for p in range(min(3, len(doc))))
        out.append({
            "file": fn,
            "doc_meta": doc.metadata,
            "first_pages_text": text[:1000]  # truncate long text
        })
    return out

def store_metadata(payload: Dict) -> str:
    """Persist discovery output to disk so next agent can load it."""
    with open("discovery_output.json", "w") as f:
        json.dump(payload, f, indent=2)
    return "✅ discovery_output.json written"

# ────────────  AGENT  ────────────
model = OpenAIChatCompletionClient(model="gemini-1.5-flash",api_key="")
discovery_agent = AssistantAgent(
    name="discoverer",
    system_message="Analyse metadata from  PDFs and Postgres and give insights in json format", 
    model_client=model,
    tools=[
       # FunctionTool(extract_db_metadata,description="Extract metadata from Postgres"),
        FunctionTool(extract_pdf_metadata,description="Extract metadata from PDFs"),
        FunctionTool(store_metadata,description="Store metadata to disk")
    ],
    reflect_on_tool_use=True,
)
