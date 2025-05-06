"""Kick‑off script that orchestrates the three AutoGen agents in a round‑robin loop."""
import asyncio, json
from autogen_agentchat.teams import RoundRobinGroupChat

from agents.metadata_discovery_autogen import discovery_agent
from agents.metadata_enrichment_autogen import enrichment_agent
from agents.relationship_autogen import relationship_agent

async def main():
    team = RoundRobinGroupChat(
        participants=[discovery_agent, enrichment_agent, relationship_agent],
        max_turns=2,
    )
    await team.run(task="Execute full metadata pipeline: discover → enrich → relate.")

if __name__ == "__main__":
    asyncio.run(main())
