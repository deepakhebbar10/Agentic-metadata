#!/usr/bin/env python3
import argparse
import json

# 1) Chain machinery
from langchain_core.runnables import RunnableLambda
from langchain_core.runnables.base import RunnableSequence

# 2) Your S3-enabled agents
from langchain_agent_embedings_v5 import (
    MetadataDiscoveryAgent,
    MetadataEnrichmentAgent,
    RelationshipDiscoveryAgent,
)

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
    parser.add_argument("--prefix", default="", help="S3 prefix (folder/) under which your files live")
    args = parser.parse_args()

    # Build & invoke the chain
    chain = create_metadata_chain(bucket=args.bucket, prefix=args.prefix)
    result = chain.invoke({})

    # Print final combined output
    print("\n===== Final Output =====")
    print(json.dumps(result, indent=2, ensure_ascii=False))

    # Print the ASCII graph
    print("\n===== Chain Graph =====")
    chain.get_graph().print_ascii()
