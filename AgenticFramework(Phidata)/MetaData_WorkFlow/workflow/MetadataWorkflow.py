from phi.workflow import Workflow
from typing import Optional
from pydantic import Field
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))
from Agents.MetaDataDiscoveryAgent import MetaDataDiscoveryAgent
from Agents.MetaDataEnrichmentAgent import MetaDataEnrichmentAgent
from Agents.MetaDataRelationshipAgent import MetaDataRelationshipAgent
from globals import NEW_FILES_FOUND
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
        self.discovery_agent.run_agent(bucket_name, prefix)

        
        # Step 2: Enrich metadata
        print("Enriching metadata...")
        self.enrichment_agent.run_agent("metadata_discovery.json", "metadata_enriched.json")

        # Step 3: Find relationships
        print("Finding metadata relationships...") 
        self.relationship_agent.run_agent("metadata_enriched.json","metadata_relationships.json")
        print("Metadata relationships analysis completed.")
        print("Workflow Completed")
        