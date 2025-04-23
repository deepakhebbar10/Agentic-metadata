import json
import psycopg2
import ollama  # Using Mistral via Ollama

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

class RelationshipDiscoveryAgent:
    def __init__(self, db_config):
        self.db_config = db_config
        self.metadata_relationships = {}  # Stores discovered relationships

    def connect_to_db(self):
        """Connect to PostgreSQL database."""
        try:
            conn = psycopg2.connect(**self.db_config)
            return conn
        except Exception as e:
            print(f"❌ Database Connection Error: {e}")
            return None

    def analyze_relationships_with_mistral(self, column, description):
        """Use Mistral LLM to analyze metadata relationships."""
        prompt = f"""
        You are an AI assistant analyzing metadata relationships.
        Given the metadata field `{column}` with the following enriched description:
        
        "{description}"
        
        Explain its relationship with other metadata fields in a structured database.
        """
        response = ollama.chat("mistral", messages=[{"role": "user", "content": prompt}])
        return response['message'] if isinstance(response, dict) and 'message' in response else str(response)

    def extract_metadata_relationships(self):
        """Retrieve relationships from `metadata_enriched` and analyze with Mistral."""
        conn = self.connect_to_db()
        if not conn:
            return None

        cursor = conn.cursor()

        # Query enriched metadata from `metadata_enriched`
        cursor.execute("SELECT source, enriched_description FROM metadata_enriched;")
        enriched_metadata = cursor.fetchall()
        
        relationships = {}
        for source, description in enriched_metadata:
            relationships[source] = self.analyze_relationships_with_mistral(source, description)

        self.metadata_relationships = relationships
        conn.close()

    def store_relationships_in_db(self):
       """Store all discovered metadata relationships in PostgreSQL."""
       conn = self.connect_to_db()
       if not conn:
           return

       cursor = conn.cursor()

    # Ensure relationships table exists with a UNIQUE constraint on `source`
       cursor.execute("""
           CREATE TABLE IF NOT EXISTS metadata_relationships (
              id SERIAL PRIMARY KEY,
               source TEXT UNIQUE,  -- Ensure source column is unique
               relationships JSONB
           );
      """)

    # Store each category separately
       for source, relationships in self.metadata_relationships.items():
        cursor.execute("""
            INSERT INTO metadata_relationships (source, relationships)
            VALUES (%s, %s)
            ON CONFLICT (source) DO UPDATE SET relationships = EXCLUDED.relationships;
        """, (source, json.dumps(relationships)))

       conn.commit()
       conn.close()
       print("✅ All metadata relationships stored successfully in PostgreSQL (`metadata_relationships`).")


    def save_relationships_to_json(self):
        """Save discovered metadata relationships as JSON backup."""
        with open("metadata_relationships.json", "w") as f:
            json.dump(self.metadata_relationships, f, indent=4)
        print("✅ Metadata relationships saved successfully to `metadata_relationships.json`.")

    def discover_relationships(self):
        """Run the relationship discovery process with Mistral."""
        print("🔍 Retrieving relationships from `metadata_enriched` and analyzing with Mistral...")
        self.extract_metadata_relationships()

        print("💾 Storing discovered relationships...")
        self.save_relationships_to_json()
        self.store_relationships_in_db()

        return self.metadata_relationships


# ---------------------- #
# 🚀 Run Relationship Discovery Agent
# ---------------------- #

if __name__ == "__main__":
    agent = RelationshipDiscoveryAgent(DB_CONFIG)
    discovered_relationships = agent.discover_relationships()
    print(json.dumps(discovered_relationships, indent=4))  # Print discovered relationships for verification
