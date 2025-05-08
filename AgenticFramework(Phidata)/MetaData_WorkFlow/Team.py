from phi.agent import Agent

from Agents.MetaDataDiscoveryAgent import MetaDataDiscoveryAgent
from Agents.MetaDataEnrichmentAgent import MetaDataEnrichmentAgent
from Agents.MetaDataRelationshipAgent import MetaDataRelationshipAgent
from phi.model.ollama import Ollama
MetaDataAgent = Agent(
model=Ollama(id="mistral:latest"),
    name="MetaDataAgent",
    description="A multi-agent system for metadata discovery, enrichment, and relationship analysis.",
    team=[MetaDataDiscoveryAgent(),MetaDataEnrichmentAgent(),MetaDataRelationshipAgent()],
    team_response_separator="=======================",
    team_response_format="json"

)
    

MetaDataAgent.run("Hi Extract metadata from the files in the S3 bucket and enrich it with additional insights and suggesstions and find relationships between the metadata of the files")

