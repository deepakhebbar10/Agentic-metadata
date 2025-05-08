import streamlit as st
from pymongo import MongoClient

# Connect to MongoDB
client = MongoClient("mongodb://localhost:27017/")
db = client.handle_db
collection = db.documents

st.title("📂 HANDLE Metadata Explorer (PDFs)")

# Filter zone
zone_filter = st.selectbox("Filter by Data Zone", ["All", "raw", "curated", "processed"])
query = {} if zone_filter == "All" else {"zone": zone_filter}

# Search by metadata key/value
key_filter = st.text_input("Search Metadata Key")
val_filter = st.text_input("Search Metadata Value")

if key_filter and val_filter:
    query["metadata.properties"] = {
        "$elemMatch": {"key": {"$regex": key_filter, "$options": "i"},
                       "value": {"$regex": val_filter, "$options": "i"}}
    }

# Retrieve and display matching documents
docs = list(collection.find(query))

st.markdown(f"### 🔍 {len(docs)} PDF(s) Found")

for doc in docs:
    with st.expander(f"📄 {doc['_id']}"):
        st.text(f"Path: {doc['path']}")
        st.text(f"Zone: {doc['zone']}")
        for m in doc.get("metadata", []):
            st.markdown(f"**🧠 Metadata Type: {m['type']}**")
            for prop in m.get("properties", []):
                st.markdown(f"- `{prop['key']}`: {prop['value']}")

