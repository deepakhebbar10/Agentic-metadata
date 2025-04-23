import json
import psycopg2
import ollama  # Using Mistral LLM
import os

# ---------------------- #
# 📂 Database Connection
# ---------------------- #

DB_CONFIG = {
    "dbname": "metadata",
    "user": "postgres",
    "password": "deepak",
    "host": "localhost",  # Change if needed
    "port": "5432"
}

class MetadataEnrichmentAgent:
    def __init__(self, db_config):
        self.db_config = db_config
        self.metadata = {}  # Stores enriched metadata

    def connect_to_db(self):
        """Connect to PostgreSQL database."""
        try:
            conn = psycopg2.connect(**self.db_config)
            return conn
        except Exception as e:
            print(f"❌ Database Connection Error: {e}")
            return None

    def fetch_raw_metadata(self):
        """Retrieve metadata from PostgreSQL (`metadata_table`)."""
        conn = self.connect_to_db()
        if not conn:
            return None

        cursor = conn.cursor()

        # Fetch raw metadata
        cursor.execute("SELECT * FROM metadata_table;")
        raw_metadata = cursor.fetchall()
        column_names = [desc[0] for desc in cursor.description]

        conn.close()
        return [dict(zip(column_names, row)) for row in raw_metadata]

    def enrich_metadata_with_mistral(self, metadata_entry):
        """Use Mistral LLM to enhance metadata with descriptions & semantic labels."""
        prompt = f"""
        Given the following metadata field: {metadata_entry['metadata']}
        - Generate a meaningful description.
        - Identify its semantic category (e.g., "Price", "Review Data", "Product Details").
        - Suggest any missing values.
        """
        response = ollama.chat("mistral", messages=[{"role": "user", "content": prompt}])
        return response['message'] if isinstance(response, dict) and 'message' in response else str(response)

    def process_metadata(self):
        """Enrich metadata and store results."""
        raw_metadata = self.fetch_raw_metadata()
        if not raw_metadata:
            print("❌ No raw metadata found!")
            return

        enriched_metadata_list = []
        for entry in raw_metadata:
            enriched_text = self.enrich_metadata_with_mistral(entry)
            enriched_entry = {
                "source": entry["source"],
                "original_metadata": entry["metadata"],
                "enriched_description": enriched_text
            }
            enriched_metadata_list.append(enriched_entry)

        self.metadata["Enriched_Metadata"] = enriched_metadata_list

    def store_enriched_metadata(self):
        """Store enriched metadata in PostgreSQL (`metadata_enriched` table)."""
        conn = self.connect_to_db()
        if not conn:
            return

        cursor = conn.cursor()

        # Ensure enriched metadata table exists
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS metadata_enriched (
                id SERIAL PRIMARY KEY,
                source TEXT,
                original_metadata JSONB,
                enriched_description TEXT
            );
        """)

        # Insert enriched metadata
        for meta in self.metadata["Enriched_Metadata"]:
            cursor.execute("""
                INSERT INTO metadata_enriched (source, original_metadata, enriched_description) 
                VALUES (%s, %s, %s);
            """, (meta["source"], json.dumps(meta["original_metadata"]), meta["enriched_description"]))

        conn.commit()
        conn.close()
        print("✅ Enriched metadata stored successfully in PostgreSQL (`metadata_enriched`).")

    def save_metadata_to_json(self):
        """Save enriched metadata as JSON backup."""
        with open("metadata_enriched.json", "w") as f:
            json.dump(self.metadata, f, indent=4)
        print("✅ Enriched metadata saved successfully to `metadata_enriched.json`.")

    def enrich_metadata(self):
        """Run the full enrichment process."""
        print("📡 Fetching metadata from PostgreSQL...")
        self.process_metadata()

        print("💾 Storing enriched metadata...")
        self.save_metadata_to_json()
        self.store_enriched_metadata()

        return self.metadata


# ---------------------- #
# 🚀 Run Metadata Enrichment Agent
# ---------------------- #

if __name__ == "__main__":
    agent = MetadataEnrichmentAgent(DB_CONFIG)
    enriched_metadata = agent.enrich_metadata()
    print(json.dumps(enriched_metadata, indent=4))  # Print enriched metadata for verification
