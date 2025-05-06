# run_pipeline.py – manual, zero-surprise driver
import json
import time
from agents.jsonhelper import parse_llm_blob
from agents.metadata_discovery_autogen import (
    extract_db_metadata,
    extract_pdf_metadata,
    store_metadata,
)
from agents.metadata_enrichment_autogen import (
    load_discovery,
    enrich,
    store_enriched,
)
from agents.relationship_autogen import (
    load_enriched,
    infer_relationships,
    store_rels,
)
from config import DB_CONFIG, PDF_DIR

def main():
    # ── 1. DISCOVERY ────────────────────────────────────────────────
    payload = {
            "pdfs": extract_pdf_metadata(PDF_DIR),
            }
    print(store_metadata(payload))           # ➜ discovery_output.json

        # ── 2. ENRICHMENT ──────────────────────────────────────────────
    raw = load_discovery()
    enriched=[]
    enriched = [enrich(entry) | entry for entry in raw["pdfs"]]
    for entry in raw["pdfs"]:
            time.sleep(30)
            enriched.append(enrich(entry))
            
    print(store_enriched(enriched))          # ➜ enriched_output.json

    # ── 3. RELATIONSHIP-MINING ─────────────────────────────────────
    rels = infer_relationships(load_enriched())
    print(store_rels(rels[0]["comment"]))      # ➜ relationships_output.json

if __name__ == "__main__":
    main()
