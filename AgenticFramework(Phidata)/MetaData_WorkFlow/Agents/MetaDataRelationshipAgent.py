from phi.agent import Agent
from phi.model.openai import OpenAIChat
import json
from typing import List, Dict
import requests
from phi.tools import tool
from phi.model.ollama import Ollama
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

@tool
def find_relationships(input_file:str,output_file:str) -> Dict:
        """Find relationships between multiple metadata records"""
        print("Tool is called by the Agent")
        with open(input_file, 'r') as f:
            metadata_list = json.load(f)
        relationships=[]
        for i,md in enumerate(metadata_list):
            others = metadata_list[:i]+metadata_list[i+1:]

            prompt = f"""
You are a metadata-analysis AI.

The target field is ⁠ {md} ⁠, with the enriched description:



Below are other fields (with their file or DB names):

{others}

Please:
1.⁠ ⁠Determine which of these fields are related to ⁠ {md} ⁠.
2.⁠ ⁠For each related field, state the file or database where it appears.
3.⁠ ⁠In brief describe the nature of the relationship based on the content of the fields.
   (e.g., same product, similar price range, related product to same comapny, shared discount pattern).


Note:it should not give relationship because it belongs to amazon invoices there should be actual relationship present like
 same customer is there in 2 invoice or they the invoices have the same product being bought etc.
 then the relationship should be established 

Example of how the relationship should be:
Scenario:invoice112 file and invoice114 file have the same product  MI Usb Type-C Cable Smartphone.
description should be like Shares the product MI Usb Type-C Cable Smartphone.
   Both files identify the same product.
 
Example of how the relationship should not be:Shares the characteristic of analyzing Amazon purchase invoices. 
   Both identify key information like product name, prices, discounts, and customer details.

the above scenario are just examples and not the actual relationship. so you should form similar 
relationships the relationship need not be just on product it can be on the same customer name 
or same discount pattern etc.

Return valid JSON where each key is a related field name, and its value 
is an object with:
•⁠  ⁠"file": the source file or database 
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
                print(f"Relationship analysis for  done.")
            else:
                print(f"Error: {response.status_code} - {response.text}")
                relationships.append("Error generating insights")


                # Save the relationship analysis
        print("Relationships finding done")
        with open(output_file, 'w') as f:
            json.dump(relationships, f, indent=2)



class MetaDataRelationshipAgent(Agent):
    def __init__(self):
        super().__init__(
            name="MetaDataRelationshipAgent",
            model=Ollama(id="mistral:latest",temperature=0.3,  # Lower = more deterministic
                max_tokens=1000,
                tool_choice="auto",  # Let model decide when to use tools
                response_format={"type": "json_object"}    
                         ),
            description="You are an Assistant to help Identify relationships between different files' metadata",
            show_tool_calls=True,markdown=True,debug_mode=True,
            tools=[find_relationships],
            system_prompt=""""
            You MUST follow these rules:
1. When asked to find rlationships from  metadata, ALWAYS use the find_relationships tool
2. NEVER describe how to use the tool - just use it
3. Dont Describe the steps you are taking
4. Do not include any other information in your response
5.Just Use the tool for completing the task.   
            """)
    
    
    
    def run_agent(self, input_file: str = 'metadata_enriched.json', output_file: str = 'metadata_relationships.json'):
        """Run the relationship analysis process"""
        
        self.run("Just call the find_relationships tool to find relatonships between metadata from input file"+input_file+"and save the output to "+output_file)