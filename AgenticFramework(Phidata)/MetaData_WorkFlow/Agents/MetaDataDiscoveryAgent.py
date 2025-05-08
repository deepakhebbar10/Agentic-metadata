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
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from phi.tools import tool
from typing import ClassVar
from phi.tools import tool
from globals import NEW_FILES_FOUND

@tool
def discover_metadata(bucket_name: str) -> List[Dict]:
        """
    Discover metadata for files in an S3 bucket.

    Args:
        bucket_name (str): The name of the S3 bucket.


    Returns:
        str: A structured JSON string with metadata for each file.
    """
        print("Discovering MetaData....")



        metadata_list = []
        prefix = ''  # Specify the prefix if needed


        try:
            with open("processed_files.txt", 'r') as f:
                processed_files = set(json.load(f))
        except (FileNotFoundError, json.JSONDecodeError):
            processed_files = set()
        # Initialize the S3 client
        no_of_files=len(processed_files)
        s3_client = boto3.client('s3')
        paginator = s3_client.get_paginator('list_objects_v2')
        for page in paginator.paginate(Bucket=bucket_name, Prefix=prefix):
            for obj in page.get('Contents', []):
                if obj['Key'] in processed_files:
                    print(f"Skipping already processed file: {obj['Key']}")
                    continue
                # Skip files that are not PDFs
                if not obj['Key'].endswith('.pdf'):
                    print(f"Skipping non-PDF file: {obj['Key']}")
                    continue
                # Skip files that are not in the specified prefix
                if prefix and not obj['Key'].startswith(prefix):
                    print(f"Skipping file not in prefix: {obj['Key']}")
                    continue
                # Extract metadata from the file
                file_metadata = get_file_metadata(bucket_name, obj['Key'],s3_client)
                processed_files.add(obj['Key'])
                metadata_list.append(file_metadata)
        if(no_of_files==len(processed_files)):

            print("No new files found to process.")
        else:
            global NEW_FILES_FOUND
            NEW_FILES_FOUND=True
            print(NEW_FILES_FOUND)
            print(f"Processed {len(processed_files)-no_of_files} new files.")
        
            print(metadata_list)
            output_file = 'metadata_discovery.json'
            # Save the metadata to a file
            with open("processed_files.txt", 'w') as f:
                json.dump(list(processed_files), f, indent=2)
            with open(output_file, 'w') as f:
                json.dump(metadata_list, f, indent=2)
            print(f"Metadata saved to {output_file}")

        print("Metadata discovery completed.")


        return json.dumps(metadata_list, indent=2)

def get_file_metadata(bucket_name: str, file_key: str,s3_client) -> Dict:
        """Extract basic metadata from an S3 file"""
        print(f"Extracting metadata for {file_key}...")
        response = s3_client.get_object(Bucket=bucket_name, Key=file_key)
        content=PyPDF2.PdfReader(BytesIO(response['Body'].read()))
        text = [page.extract_text() for page in content.pages]
        text = "\n".join(text)
        metadata = {
            'file_name': file_key,
            'content_type': response.get('ContentType', 'unknown'),
            'last_modified': str(response['LastModified']),
            's3_uri': f"s3://{bucket_name}/{file_key}",
            "insights": generate_insights(text)
        }
        return metadata

def generate_insights(content: str) -> str:
        prompt="""
    Analyse the given content and generate key insights of the file: 
"""+content
        payload = {
            "model": "mistral:latest",  # or "mistral:latest"
            "prompt": prompt,
            "stream": False  # Set to True if your endpoint supports streaming
        }
        print("Generating insights on file content...")
        reponse = requests.post("http://localhost:11434/api/generate",json=payload)
        if reponse.status_code == 200:
            response_json = reponse.json()
            return response_json["response"]
        else:
            print(f"Error: {reponse.status_code} - {reponse.text}")
            return "Error generating insights"

class MetaDataDiscoveryAgent(Agent):
    def __init__(self):
        super().__init__(
            name="MetaDataDiscoveryAgent",
            model=Ollama(id="mistral:latest",
                         
                         temperature=0.3,  # Lower = more deterministic
                max_tokens=1000,
                tool_choice="auto",  # Let model decide when to use tools
                response_format={"type": "json_object"}
                         
                         ),
            description="You are an Assistant to discover metadata from files in S3 buckets. You must use the discover_metadata tool to extract metadata from files",
            tools=[discover_metadata],
            show_tool_calls=True,markdown=True,debug_mode=True,
            system_prompt="""You MUST follow these rules:
1. When asked to discover metadata, ALWAYS use the discover_metadata tool
2. NEVER describe how to use the tool - just use it
3. Dont Describe the steps you are taking
4. Do not include any other information in your response
5. Return ONLY the tool's raw output
6. Use JSON format for all responses"""
        )
        self.output_file="metadata_discovery.json"
        self.bucket_name = None
        self.prefix = None
    
    def run_agent(self, bucket_name: str, prefix: str = '', output_file: str = 'metadata_discovery.json'):
        """Run the metadata discovery process"""
        self.bucket_name = bucket_name
        self.prefix = prefix
        self.output_file = output_file
        metadata : RunResponse = self.run("Just call the discover_metadata tool with bucket_name set to 'datalake-ai-agents'. Return only the tools output")
        print("Metadata discovery completed.")