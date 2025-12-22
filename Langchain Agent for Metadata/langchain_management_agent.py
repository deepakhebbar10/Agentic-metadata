#!/usr/bin/env python3
import argparse
import json
import psycopg2
import os
import google.generativeai as genai
import ollama

from langchain_core.runnables import RunnableLambda
from langchain_core.runnables.base import RunnableSequence

from langchain_agent_embedings_v7 import (
    MetadataDiscoveryAgent,
    MetadataEnrichmentAgent,
    RelationshipDiscoveryAgent,
    list_s3_keys,
    SYNC_DB_URI,
)

# ── Gemini helper & LLM wrapper ────────────────────────────────────────────────
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
    """Simple wrapper: Gemini → Ollama fallback."""
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


# ── New-file detection ──────────────────────────────────────────────────────────
def detect_new_files(bucket: str, prefix: str):
    all_keys = list_s3_keys(bucket, prefix)
    with psycopg2.connect(SYNC_DB_URI) as conn, conn.cursor() as cur:
        cur.execute("SELECT source FROM metadata_langchain_table;")
        seen = {r[0] for r in cur.fetchall()}
    return [k for k in all_keys if k not in seen]


# ── Chain construction ─────────────────────────────────────────────────────────
def detect_new_files(bucket: str, prefix: str):
    all_keys = list_s3_keys(bucket, prefix)
    with psycopg2.connect(SYNC_DB_URI) as conn, conn.cursor() as cur:
        cur.execute("SELECT source FROM metadata_langchain_table;")
        seen = {r[0] for r in cur.fetchall()}
    return [k for k in all_keys if k not in seen]

def create_metadata_chain(bucket: str, prefix: str) -> RunnableSequence:
    # Instantiate each agent with S3 info on the discovery step
    discover_agent   = MetadataDiscoveryAgent(use_gemini=True, bucket=bucket, prefix=prefix)
    enrich_agent     = MetadataEnrichmentAgent(use_gemini=True)
    relationship_agent = RelationshipDiscoveryAgent(use_gemini=True)

    # Wrap them in RunnableLambdas
    discovery_step = RunnableLambda(
        lambda _: {"raw_meta": discover_agent.extract_metadata()},
        name="discovery_step",
    )
    enrichment_step = RunnableLambda(
        lambda ctx: {**ctx, "enriched_meta": enrich_agent.enrich_metadata()},
        name="enrichment_step",
    )
    relationship_step = RunnableLambda(
        lambda ctx: {**ctx, "relations": relationship_agent.discover_relationships()},
        name="relationship_step",
    )

    # Chain them together
    return discovery_step | enrichment_step | relationship_step

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run full metadata pipeline via S3 + chain")
    parser.add_argument("--bucket", required=True, help="Name of your S3 bucket")
    parser.add_argument("--prefix", default="", help="S3 prefix under which your files live")
    args = parser.parse_args()

    new_keys = detect_new_files(args.bucket, args.prefix)
    if not new_keys:
        print("✅ No new files in S3. Nothing to do.")
        exit(0)

    # LLM notification (optional)
    llm = AgentLLM(use_gemini=True)
    notice = llm.analyze(
        f"Detected {len(new_keys)} new file(s):\n" +
        "\n".join(f"- {k}" for k in new_keys) +
        "\nStarting metadata extraction."
    )
    print("🔔 LLM notification:\n", notice)

    print('calling the chain now')
    chain  = create_metadata_chain(bucket=args.bucket, prefix=args.prefix)
    result = chain.invoke({"new_files": new_keys})

    print("\n===== Final Output =====")
    print(json.dumps(result, indent=2, ensure_ascii=False))

    print("\n===== Chain Graph =====")
    chain.get_graph().print_ascii()