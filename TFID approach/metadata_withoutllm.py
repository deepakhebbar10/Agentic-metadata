import fitz  # PyMuPDF for PDFs
import json
import os
import psycopg2
import pdfplumber
import spacy
import re
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer

# Load NLP model for Named Entity Recognition (NER)
nlp = spacy.load("en_core_web_sm")

# ---------------------- #
# 📂 Database Connection
# ---------------------- #

DB_CONFIG = {
    "dbname": "metadata",
    "user": "postgres",
    "password": "deepak",
    "host": "localhost",
    "port": "5432"
}

class MetadataProcessingWithoutLLM:
    def __init__(self, db_config, pdf_dir="C:/Users/deepa/Desktop/Minor Project Meta Data Agent/data/"):
        self.db_config = db_config
        self.pdf_dir = pdf_dir
        self.metadata = {}
        self.enriched_metadata = {}
        self.relationships = {}

    def connect_to_db(self):
        """Connect to PostgreSQL database."""
        try:
            conn = psycopg2.connect(**self.db_config)
            return conn
        except Exception as e:
            print(f"❌ Database Connection Error: {e}")
            return None

    # ---------------------- #
    # 📡 Metadata Extraction (Database & PDFs)
    # ---------------------- #
    
    def extract_db_metadata(self):
        """Extract metadata from PostgreSQL database."""
        conn = self.connect_to_db()
        if not conn:
            return

        cursor = conn.cursor()
        cursor.execute("""
            SELECT column_name, data_type FROM information_schema.columns WHERE table_name = 'product_reviews';
        """)
        columns = cursor.fetchall()
        self.metadata["DB_Columns"] = [{"name": col[0], "type": col[1]} for col in columns]

        cursor.execute("SELECT COUNT(*) FROM product_reviews;")
        self.metadata["DB_Row_Count"] = cursor.fetchone()[0]

        conn.close()

    def extract_pdf_metadata(self):
        """Extract metadata from all PDFs in the directory."""
        pdf_metadata_list = []
        for pdf_file in os.listdir(self.pdf_dir):
            if pdf_file.endswith(".pdf"):
                pdf_path = os.path.join(self.pdf_dir, pdf_file)
                doc = fitz.open(pdf_path)
                pdf_meta = doc.metadata

                with pdfplumber.open(pdf_path) as pdf:
                    text_content = "\n".join(page.extract_text() for page in pdf.pages if page.extract_text())

                # Extract keywords using TF-IDF
                vectorizer = TfidfVectorizer(stop_words="english", max_features=10)
                tfidf_matrix = vectorizer.fit_transform([text_content])
                keywords = vectorizer.get_feature_names_out()

                pdf_metadata_list.append({
                    "File Name": pdf_file,
                    "Title": pdf_meta.get("title", "Unknown"),
                    "Author": pdf_meta.get("author", "Unknown"),
                    "Extracted Keywords": list(keywords)
                })

        self.metadata["PDF_Metadata"] = pdf_metadata_list

    # ---------------------- #
    # 🏗️ Metadata Enrichment (NER & Rules)
    # ---------------------- #

    def enrich_metadata(self):
        """Perform Named Entity Recognition (NER) and manual enrichment."""
        self.enriched_metadata["DB_Metadata"] = []
        self.enriched_metadata["PDF_Metadata"] = []

        for column in self.metadata.get("DB_Columns", []):
            column_name = column["name"]
            column_type = column["type"]

            # Manually enrich column descriptions
            description = "This is a text field." if column_type in ["text", "varchar"] else "This is a numerical field."
            if "id" in column_name.lower():
                description = "Likely a primary or foreign key."
            elif "price" in column_name.lower():
                description = "Stores pricing information."
            elif "date" in column_name.lower():
                description = "Represents a timestamp or event date."

            self.enriched_metadata["DB_Metadata"].append({
                "column_name": column_name,
                "description": description
            })

        for pdf_data in self.metadata.get("PDF_Metadata", []):
            doc = nlp(" ".join(pdf_data.get("Extracted Keywords", [])))
            named_entities = {ent.label_: ent.text for ent in doc.ents}

            self.enriched_metadata["PDF_Metadata"].append({
                "file_name": pdf_data["File Name"],
                "entities": named_entities
            })

    # ---------------------- #
    # 🔗 Metadata Relationship Discovery
    # ---------------------- #

    def discover_relationships(self):
        """Find relationships between metadata fields using heuristic rules."""
        self.relationships = defaultdict(list)

        # Rule-based relationships for DB columns
        for column in self.metadata.get("DB_Columns", []):
            column_name = column["name"]
            if "id" in column_name.lower():
                self.relationships[column_name].append("Likely a primary or foreign key.")
            if "price" in column_name.lower():
                self.relationships[column_name].append("Linked to product pricing.")
            if "date" in column_name.lower():
                self.relationships[column_name].append("Represents a timestamp.")

        # Link PDF metadata with database metadata
        for pdf_meta in self.metadata.get("PDF_Metadata", []):
            file_name = pdf_meta["File Name"]
            extracted_keywords = pdf_meta.get("Extracted Keywords", [])

            for keyword in extracted_keywords:
                for column in self.metadata.get("DB_Columns", []):
                    if keyword.lower() in column["name"].lower():
                        self.relationships[file_name].append(f"Related to `{column['name']}` in database.")

    # ---------------------- #
    # 💾 Storage (JSON & Database)
    # ---------------------- #

    def save_metadata_to_json(self):
        """Save extracted metadata, enriched metadata, and relationships separately."""
        with open("metadata_without_llm.json", "w") as f:
            json.dump(self.metadata, f, indent=4)

        with open("metadata_enriched_without_llm.json", "w") as f:
            json.dump(self.enriched_metadata, f, indent=4)

        with open("metadata_relationships_without_llm.json", "w") as f:
            json.dump(self.relationships, f, indent=4)

        print("✅ Metadata saved successfully as JSON.")

    def store_metadata_in_db(self):
        """Store extracted metadata in PostgreSQL in separate tables."""
        conn = self.connect_to_db()
        if not conn:
            return

        cursor = conn.cursor()
        cursor.execute("CREATE TABLE IF NOT EXISTS metadata_without_llm (id SERIAL PRIMARY KEY, source TEXT, metadata JSONB);")
        cursor.execute("CREATE TABLE IF NOT EXISTS metadata_enriched_without_llm (id SERIAL PRIMARY KEY, source TEXT, enriched_metadata JSONB);")
        cursor.execute("CREATE TABLE IF NOT EXISTS metadata_relationships_without_llm (id SERIAL PRIMARY KEY, source TEXT, relationships JSONB);")

        cursor.execute("INSERT INTO metadata_without_llm (source, metadata) VALUES (%s, %s);", ("Database", json.dumps(self.metadata)))
        cursor.execute("INSERT INTO metadata_enriched_without_llm (source, enriched_metadata) VALUES (%s, %s);", ("Database", json.dumps(self.enriched_metadata)))
        cursor.execute("INSERT INTO metadata_relationships_without_llm (source, relationships) VALUES (%s, %s);", ("Database", json.dumps(self.relationships)))

        conn.commit()
        conn.close()
        print("✅ Metadata stored successfully in PostgreSQL.")

# ---------------------- #
# 🚀 Run the Agent (Without LLM)
# ---------------------- #
if __name__ == "__main__":
    agent = MetadataProcessingWithoutLLM(DB_CONFIG)
    agent.extract_db_metadata()
    agent.extract_pdf_metadata()
    agent.enrich_metadata()
    agent.discover_relationships()
    agent.save_metadata_to_json()
    agent.store_metadata_in_db()
