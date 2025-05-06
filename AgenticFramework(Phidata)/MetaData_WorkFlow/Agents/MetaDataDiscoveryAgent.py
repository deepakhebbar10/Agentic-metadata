from phi.agent import Agent
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
class MetaDataDiscoveryAgent(Agent):
    discover_metadata: ClassVar[tool] = tool
    def __init__(self):
        super().__init__(
            name="MetaDataDiscoveryAgent",
            llm=OpenAIChat(model="gpt-4"),
            description="You are an Assistant to discovers metadata from files in S3 buckets. You can use the tools to extract metadata from files",
            tools=[self.discover_metadata],
            show_tool_calls=True,markdown=True,debug_mode=True
        )
        self.output_file="metadata_discovery.json"
        self.bucket_name = None
        self.prefix = None
        self.s3_client = boto3.client('s3')
    
    def generate_insights(self, content: str) -> str:
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
        
    def get_file_metadata(self, bucket_name: str, file_key: str) -> Dict:
        """Extract basic metadata from an S3 file"""
        response = self.s3_client.get_object(Bucket=bucket_name, Key=file_key)
        content=PyPDF2.PdfReader(BytesIO(response['Body'].read()))
        text = [page.extract_text() for page in content.pages]
        text = "\n".join(text)
        metadata = {
            'file_name': file_key,
            'content_type': response.get('ContentType', 'unknown'),
            'last_modified': str(response['LastModified']),
            's3_uri': f"s3://{bucket_name}/{file_key}",
            "insights":self.generate_insights(text)
        }
        return metadata
    def discover_metadata(self, bucket_name: str, prefix: str = '') -> List[Dict]:
        """Discover metadata for all files in an S3 bucket path"""
        metadata_list = []
        
        paginator = self.s3_client.get_paginator('list_objects_v2')
        for page in paginator.paginate(Bucket=bucket_name, Prefix=prefix):
            for obj in page.get('Contents', []):
                file_metadata = self.get_file_metadata(bucket_name, obj['Key'])
                metadata_list.append(file_metadata)
        
        return metadata_list
    
    def run(self, bucket_name: str, prefix: str = '', output_file: str = 'metadata_discovery.json'):
        """Run the metadata discovery process"""
        self.bucket_name = bucket_name
        self.prefix = prefix
        self.output_file = output_file
        metadata = self.discover_metadata(bucket_name,prefix)
        
        # Save the discovered metadata
        with open(output_file, 'w') as f:
            json.dump(metadata, f, indent=2)