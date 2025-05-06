from phi.agent import Agent
from phi.model.openai import OpenAIChat
import json
from typing import List, Dict
import requests
class MetaDataEnrichmentAgent(Agent):
    def __init__(self):
        super().__init__(
            name="MetaDataEnrichmentAgent",
            llm=OpenAIChat(model="gpt-4"),
            description="Enriches metadata with additional insights and suggestions"
        )
    
    def enrich_metadata(self, metadata: Dict) -> Dict:
        """Enrich a single metadata record"""
        prompt = f"""
        Analyze this file metadata and suggest additional features that could be useful:
        {json.dumps(metadata, indent=2)}
        
        Consider suggesting:
        - Potential data categories/tags
        - Data quality indicators
        - Recommended preprocessing steps
        - Potential use cases
        - Any missing metadata that would be valuable
        -Dont structure the response in a table format
        Return your response as a JSON object with:
        - original_metadata: the original input
        - suggested_features: list of suggested features
        - data_category: suggested category
        - quality_indicators: any quality notes
        """
    
        payload = {
            "model": "mistral:latest",  # or "mistral:latest"
            "prompt": prompt,
            "stream": False  # Set to True if your endpoint supports streaming
        }
        reponse = requests.post("http://localhost:11434/api/generate",json=payload)
        if reponse.status_code == 200:
            response_json = reponse.json()
            return response_json["response"]
        else:
            print(f"Error: {reponse.status_code} - {reponse.text}")
            return "Error generating insights"
    
    def run(self, input_file: str = 'metadata_discovery.json', output_file: str = 'metadata_enriched.json'):
        """Run the metadata enrichment process"""
        with open(input_file, 'r') as f:
            metadata_list = json.load(f)
        
        enriched_metadata = [self.enrich_metadata(md) for md in metadata_list]
        
        # Save the enriched metadata
        with open(output_file, 'w') as f:
            json.dump(enriched_metadata, f, indent=2)
        
        return enriched_metadata