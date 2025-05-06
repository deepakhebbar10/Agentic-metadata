from phi.agent import Agent
from phi.model.openai import OpenAIChat
import json
from typing import List, Dict
import requests
class MetaDataRelationshipAgent(Agent):
    def __init__(self):
        super().__init__(
            name="MetaDataRelationshipAgent",
            llm=OpenAIChat(model="gpt-4"),
            description="Identifies relationships between different files' metadata"
        )
    
    def find_relationships(self, metadata_list: List[Dict]) -> Dict:
        """Find relationships between multiple metadata records"""
        relationships=[]
        for i,md in enumerate(metadata_list):
            others = metadata_list[:i]+metadata_list[i+1:]

            prompt = f"""
You are a metadata-analysis AI.

The target field is ⁠ {md} ⁠, with the enriched description:



Below are other fields (with their file or DB names):

{others}


Please:
1.⁠ ⁠Determine which of these fields are related to {md} ⁠.
2.⁠ ⁠For each related field, state the file or database where it appears.
3.⁠ ⁠In brief describe the nature of the relationship 
   (e.g., same category, similar price range, related product, shared discount pattern).

Return valid JSON where each key is a related field name, and its value 
is an object with:
•⁠  ⁠"file": the source file 
•⁠  ⁠"relationship":  description 
"""
            
            payload = {
                "model": "mistral:latest",  # or "mistral:latest"
                "prompt": prompt,
                "stream": False  # Set to True if your endpoint supports streaming
            }
            response = requests.post("http://localhost:11434/api/generate",json=payload)
            if response.status_code == 200:
                response_json = response.json()
                relationships.append(response_json["response"])
            else:
                print(f"Error: {response.status_code} - {response.text}")
                relationships.append("Error generating insights")
        return relationships
    
    def run(self, input_file: str = 'metadata_enriched.json', output_file: str = 'metadata_relationships.json'):
        """Run the relationship analysis process"""
        with open(input_file, 'r') as f:
            metadata_list = json.load(f)
        
        relationships = self.find_relationships(metadata_list)
        
        # Save the relationship analysis
        with open(output_file, 'w') as f:
            json.dump(relationships, f, indent=2)
        
        return relationships