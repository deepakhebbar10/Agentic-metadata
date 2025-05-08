from phi.agent import Agent
from phi.model.openai import OpenAIChat
import json
from typing import List, Dict
import requests
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))
from phi.model.ollama import Ollama
from phi.tools import tool
from globals import NEW_FILES_FOUND
def enrich_metadata(metadata: Dict) -> Dict:
        """Enrich a single metadata record"""
        print("Enriching metadata for the file...")
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
@tool
def get_enriched_metadata(input_file:str,output_file:str) -> List[Dict]:
            

            print("Tool is Called by the Agent")
            with open(input_file, 'r') as f:
                    metadata_list = json.load(f)
            enriched_metadata = [enrich_metadata(md) for md in metadata_list]
                        # Save the enriched metadata
            print("Enriching metadata done.")
            with open(output_file, 'w') as f:
                    json.dump(enriched_metadata, f, indent=2)
            print(f"Enriched metadata saved to {output_file}")


class MetaDataEnrichmentAgent(Agent):
    def __init__(self):
        super().__init__(
            name="MetaDataEnrichmentAgent",
            model=Ollama(id="mistral:latest",
                          temperature=0.3,  # Lower = more deterministic
                max_tokens=1000,
                tool_choice="auto",  # Let model decide when to use tools
                response_format={"type": "json_object"}    
                         ),
            description="You are an Assistant to enrich the metadata.Enriches metadata with additional insights and suggestions",
            show_tool_calls=True,markdown=True,debug_mode=True,
            tools=[get_enriched_metadata],
            system_prompt=""""
            You MUST follow these rules:
1. When asked to enrich metadata, ALWAYS use the get_enriched_metadata tool
2. NEVER describe how to use the tool - just use it
3. Dont Describe the steps you are taking
4. Do not include any other information in your response
5.Just Use the tool for completing the task.   
            """
        )
    
    def run_agent(self, input_file: str = 'metadata_discovery.json', output_file: str = 'metadata_enriched.json'):
        """Run the metadata enrichment process"""

        self.run("Just call the get_enriched_metadata tool to enrich metadata from a file "+input_file+"and save it to"+output_file+".Return only the tools output")

        print(f"Enriching metadata from {input_file} and saving to {output_file}")