#!/usr/bin/env python3
from workflow.MetadataWorkflow import MetadataWorkflow  # This imports the class directly
from dotenv import load_dotenv
import os
import argparse

def main():
    load_dotenv()  # Load environment variables from .env file
    
    parser = argparse.ArgumentParser(description='Metadata Processing Workflow')
    parser.add_argument('--bucket', required=True, help='S3 bucket name')
    parser.add_argument('--prefix', default='', help='S3 prefix (optional)')
    args = parser.parse_args()
    
    # Validate environment variables
    required_env_vars = ['AWS_ACCESS_KEY_ID', 'AWS_SECRET_ACCESS_KEY', 'AWS_DEFAULT_REGION']
    for var in required_env_vars:
        if not os.getenv(var):
            raise ValueError(f"Missing required environment variable: {var}")
    
    print("Starting Metadata Processing Workflow...")
    workflow = MetadataWorkflow()  # Now this will work correctly
    
    try:
        results = workflow.run(
            bucket_name=args.bucket,
            prefix=args.prefix
        )
        
        print("\nWorkflow completed successfully!")
       # print(f"Discovered {len(results['discovered_metadata'])} files")
       # print(f"Found {len(results['relationships']['relationships'])} relationships")
        
    except Exception as e:
        print(f"\nWorkflow failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()