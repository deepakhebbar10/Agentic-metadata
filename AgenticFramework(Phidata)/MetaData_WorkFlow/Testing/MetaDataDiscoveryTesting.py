from phi.agent import Agent,RunResponse
from phi.model.openai import OpenAIChat
import boto3
import json
from typing import List, Dict, Optional
import PyPDF2
from io import BytesIO, StringIO
from textwrap import wrap
from phi.model.ollama import Ollama
import requests
from openai import OpenAI
from phi.tools import tool
from typing import ClassVar


def get_file_metadata(bucket_name: str, file_key: str,s3_client) -> Dict:
        """Extract basic metadata from an S3 file"""
        response = s3_client.get_object(Bucket=bucket_name, Key=file_key)
        content=PyPDF2.PdfReader(BytesIO(response['Body'].read()))
        text = [page.extract_text() for page in content.pages]
        text = "\n".join(text)
        metadata = {
            'file_name': file_key,
            'content_type': response.get('ContentType', 'unknown'),
            'last_modified': str(response['LastModified']),
            's3_uri': f"s3://{bucket_name}/{file_key}",
            "insights":generate_insights(text)
        }
        return metadata


@tool
def discover_metadata(bucket_name: str, prefix: str = '') -> List[Dict]:
        """
    Discover metadata for all files in the specified S3 bucket path.
    Extracts file name, content type, last modified time, S3 URI, and high-level insights.


        Args:  bucket_name (str): The name of the S3 bucket.
        prefix (str, optional): Optional prefix to filter files.

        Returns:
        List[Dict]: A list of dictionaries containing metadata for each file.
    """
        metadata_list = []
        s3_client = boto3.client('s3')
        paginator = s3_client.get_paginator('list_objects_v2')
        for page in paginator.paginate(Bucket=bucket_name, Prefix=prefix):
            for obj in page.get('Contents', []):
                file_metadata = get_file_metadata(bucket_name, obj['Key'],s3_client)
                metadata_list.append(file_metadata)
        
        return json.dumps(metadata_list)

def generate_insights(content: str) -> str:
        prompt="""
    Analyse the given content and generate key insights of the file: 
"""+content
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

class MetaDataDiscoveryAgent(Agent):
    discover_metadata: ClassVar
    def __init__(self):
        super().__init__(
            name="MetaDataDiscoveryAgent",
            model=Ollama(id="mistral:latest",
                         base_url="http://localhost:11434/api/generate",  
                        temperature=0.7,
                        type="chat",  
                        max_tokens=1024),
                         tool_choice="auto",
            description="You are an Assistant to discovers metadata from files in S3 buckets. You can use the tools to extract metadata from files",
            tools=[discover_metadata],
            show_tool_calls=True,markdown=True,debug_mode=True,
           
        )
        self.output_file="metadata_discovery.json"
        self.bucket_name = None
        self.prefix = None
        self.s3_client = boto3.client('s3')
    

    def run_agent(self, bucket_name: str, prefix: str = '', output_file: str = 'metadata_discovery1.json'):
        """Run the metadata discovery process"""
        self.bucket_name = bucket_name
        self.prefix = prefix
        self.output_file = output_file
        metadata: RunResponse = self.run("Call the `discover_metadata` tool with bucket_name set to datalake-ai-agents. Return only the tool's result.")
        # Save the discovered metadata
        print(f"Discovered metadata: {metadata}")
        with open(output_file, 'w') as f:
            json.dump(metadata.content, f, indent=2)
        return metadata.content