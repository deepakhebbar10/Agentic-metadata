import streamlit as st
import sys
from pathlib import Path

# Add root to PYTHONPATH
sys.path.append(str(Path(__file__).parent))

from workflow.MetadataWorkflow import MetadataWorkflow

# Streamlit setup
st.set_page_config(page_title="🧠 Metadata Agent Workflow", layout="centered")
st.title("🔁 Run Metadata Workflow")

# Input section
bucket_name = st.text_input("S3 Bucket Name", value="your-bucket-name")
prefix = st.text_input("Prefix (optional)", value="")

# Workflow status box
log_box = st.empty()

# Run workflow button
if st.button("🚀 Run Workflow"):
    if not bucket_name:
        st.warning("Please enter a bucket name.")
    else:
        with st.spinner("Running workflow..."):
            workflow = MetadataWorkflow()

            log_messages = []

            def log(msg):
                log_messages.append(msg)
                log_box.code("\n".join(log_messages), language="bash")

            log("✅ Starting Metadata Discovery...")
            workflow.discovery_agent.run_agent(bucket_name, prefix)
            log("✅ Discovery completed.")

            log("🔄 Running Metadata Enrichment...")
            workflow.enrichment_agent.run_agent("metadata_discovery.json", "metadata_enriched.json")
            log("✅ Enrichment completed.")

            log("🔗 Finding Metadata Relationships...")
            workflow.relationship_agent.run_agent("metadata_enriched.json", "metadata_relationships.json")
            log("✅ Relationship analysis completed.")

            log("🎉 Workflow completed.")
