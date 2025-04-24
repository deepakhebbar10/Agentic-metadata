import fitz  # PyMuPDF for PDFs
import json
import ollama
import os
import psycopg2  # PostgreSQL connector
import pandas as pd

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

class MetadataDiscoveryAgent:
    def __init__(self, db_config, pdf_dir="C:/Users/deepa/Desktop/Minor Project Meta Data Agent/data/"):
        self.db_config = db_config
        self.pdf_dir = pdf_dir
        self.metadata = {}  # Stores extracted metadata

    def connect_to_db(self):
        """Connect to PostgreSQL database."""
        try:
            conn = psycopg2.connect(**self.db_config)
            return conn
        except Exception as e:
            print(f"❌ Database Connection Error: {e}")
            return None

    def extract_db_metadata(self):
        """Extract metadata from PostgreSQL database (product_reviews table)."""
        conn = self.connect_to_db()
        if not conn:
            return

        cursor = conn.cursor()

        # Extract schema metadata
        cursor.execute("""
            SELECT column_name, data_type 
            FROM information_schema.columns 
            WHERE table_name = 'product_reviews';
        """)
        columns = cursor.fetchall()
        self.metadata["DB_Columns"] = [{"name": col[0], "type": col[1]} for col in columns]

        # Extract row count
        cursor.execute("SELECT COUNT(*) FROM product_reviews;")
        row_count = cursor.fetchone()[0]
        self.metadata["DB_Row_Count"] = row_count

        # Extract sample data 
        cursor.execute("SELECT * FROM product_reviews LIMIT 5;")
        sample_rows = cursor.fetchall()
        self.metadata["DB_Sample_Data"] = [dict(zip([col[0] for col in columns], row)) for row in sample_rows]

        conn.close()

    def extract_pdf_metadata(self):
        """Extract metadata from all PDFs in the directory."""
        pdf_metadata_list = []

        for pdf_file in os.listdir(self.pdf_dir):
            if pdf_file.endswith(".pdf"):
                pdf_path = os.path.join(self.pdf_dir, pdf_file)
                doc = fitz.open(pdf_path)
                pdf_meta = doc.metadata

                # Extract first few pages as text
                text_content = ""
                for page_num in range(min(3, len(doc))):
                    text_content += doc[page_num].get_text("text") + "\n"

                # Ask Mistral to analyze document
                insights = self.analyze_text_with_mistral(text_content)

                # Store extracted metadata
                pdf_metadata_list.append({
                    "File Name": pdf_file,
                    "File Type": "PDF",
                    "Title": pdf_meta.get("title", "Unknown"),
                    "Author": pdf_meta.get("author", "Unknown"),
                    "Creation Date": pdf_meta.get("creationDate", "Unknown"),
                    "Modification Date": pdf_meta.get("modDate", "Unknown"),
                    "Extracted Insights": insights
                })

        self.metadata["PDF_Metadata"] = pdf_metadata_list

    def analyze_text_with_mistral(self, text):
        """Use Mistral LLM to generate metadata insights from text."""
        prompt = f"Analyze this document and provide key metadata insights:\n{text}"
        response = ollama.chat("mistral", messages=[{"role": "user", "content": prompt}])
        return response['message'] if isinstance(response, dict) and 'message' in response else str(response)

    def save_metadata_to_json(self):
        """Save extracted metadata as JSON backup."""
        with open("metadata.json", "w") as f:
            json.dump(self.metadata, f, indent=4)
        print("✅ Metadata saved successfully to metadata.json")

    def store_metadata_in_db(self):
        """Store extracted metadata into PostgreSQL (metadata_table)."""
        conn = self.connect_to_db()
        if not conn:
            return

        cursor = conn.cursor()

        # Ensure metadata table exists
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS metadata_table (
                id SERIAL PRIMARY KEY,
                source TEXT,
                metadata JSONB
            );
        """)

        # Store database metadata
        cursor.execute("""
            INSERT INTO metadata_table (source, metadata) VALUES (%s, %s);
        """, ("Database", json.dumps(self.metadata["DB_Columns"])))

        # Store PDF metadata
        for pdf_meta in self.metadata["PDF_Metadata"]:
            cursor.execute("""
                INSERT INTO metadata_table (source, metadata) VALUES (%s, %s);
            """, (pdf_meta["File Name"], json.dumps(pdf_meta)))

        conn.commit()
        conn.close()
        print("✅ Metadata stored successfully in PostgreSQL (metadata_table).")

    def extract_metadata(self):
        """Extract metadata from database and PDFs, then store it."""
        print("📡 Extracting metadata from PostgreSQL...")
        self.extract_db_metadata()

        print("📄 Extracting metadata from PDFs...")
        self.extract_pdf_metadata()

        print("💾 Storing metadata...")
        self.save_metadata_to_json()
        self.store_metadata_in_db()

        return self.metadata


# ---------------------- #
# 🚀 Run Metadata Agent
# ---------------------- #

if __name__ == "__main__":
    agent = MetadataDiscoveryAgent(DB_CONFIG)
    metadata = agent.extract_metadata()
    print(json.dumps(metadata, indent=4))  # Print metadata for verification
