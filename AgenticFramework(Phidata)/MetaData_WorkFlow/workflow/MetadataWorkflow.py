from phi.workflow import Workflow
from typing import Optional
from pydantic import Field
from Agents.MetaDataDiscoveryAgent import MetaDataDiscoveryAgent
from Agents.MetaDataEnrichmentAgent import MetaDataEnrichmentAgent
from Agents.MetaDataRelationshipAgent import MetaDataRelationshipAgent

class MetadataWorkflow(Workflow):
    discovery_agent: Optional[MetaDataDiscoveryAgent] = Field(default=None, exclude=True)
    enrichment_agent: Optional[MetaDataEnrichmentAgent] = Field(default=None, exclude=True)
    relationship_agent: Optional[MetaDataRelationshipAgent] = Field(default=None, exclude=True)
    def __init__(self):
        super().__init__(
            name="MetadataProcessingWorkflow",
            description="Processes metadata from S3 through discovery, enrichment, and relationship analysis"
        )
        
        # Initialize agents
        self.discovery_agent = MetaDataDiscoveryAgent()
        self.enrichment_agent = MetaDataEnrichmentAgent()
        self.relationship_agent = MetaDataRelationshipAgent()
    
    def run(self, bucket_name: str, prefix: str = ''):
        # Step 1: Discover metadata
        print("Starting metadata discovery...")
        discovered_metadata = self.discovery_agent.run(bucket_name, prefix)
        
        # Step 2: Enrich metadata
        print("Enriching metadata...")
        enriched_metadata = self.enrichment_agent.run()
        
        # Step 3: Find relationships
        print("Finding metadata relationships...")
        relationships = self.relationship_agent.run()
        
        return {
            "discovered_metadata": discovered_metadata,
            "enriched_metadata": enriched_metadata,
            "relationships": relationships
        }