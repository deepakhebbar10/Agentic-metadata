

from phi.playground import Playground, serve_playground_app
from Agents.MetaDataDiscoveryAgent import MetaDataDiscoveryAgent
from Agents.MetaDataEnrichmentAgent import MetaDataEnrichmentAgent
from Agents.MetaDataRelationshipAgent import MetaDataRelationshipAgent
import streamlit as st

# Agent selector
agent_options = {
    "Metadata Discovery Agent": MetaDataDiscoveryAgent(),
    "Metadata Relationship Agent": MetaDataRelationshipAgent(),
    "Metadata Enrichment Agent": MetaDataEnrichmentAgent(),
}

# Streamlit UI
st.set_page_config(page_title="Agent Playground", layout="centered")
st.title("🧠 Multi-Agent Metadata RAG Interface")

agent_choice = st.selectbox("Choose an Agent:", list(agent_options.keys()))
selected_agent = agent_options[agent_choice]

query = st.text_input("Ask a question:", placeholder="e.g., What is metadata enrichment?")

if st.button("Ask"):
    if query:
        st.subheader(f"🤖 {agent_choice} says:")
        with st.spinner("Thinking..."):
            selected_agent.print_response(query, stream=True)
    else:
        st.warning("Please enter a question.")
