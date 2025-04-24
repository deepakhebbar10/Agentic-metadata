import fitz  # PyMuPDF for PDFs
import json
import os
import psycopg2
import pdfplumber
import pytesseract  # OCR for scanned PDFs
import spacy
import re
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation

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
    # 📡 Step 1: Metadata Extraction (Database & PDFs)
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
        """Extract metadata and raw text from all PDFs in the directory using pdfplumber and OCR."""
        pdf_metadata_list = []
        for pdf_file in os.listdir(self.pdf_dir):
            if pdf_file.endswith(".pdf"):
                pdf_path = os.path.join(self.pdf_dir, pdf_file)
                doc = fitz.open(pdf_path)
                pdf_meta = doc.metadata

                text_content = ""
                with pdfplumber.open(pdf_path) as pdf:
                    for page in pdf.pages:
                        extracted_text = page.extract_text()
                        if extracted_text:
                            text_content += extracted_text
                        else:
                            # Use OCR if text is not directly extractable
                            text_content += pytesseract.image_to_string(page.to_image())

                pdf_metadata_list.append({
                    "File Name": pdf_file,
                    "Title": pdf_meta.get("title", "Unknown"),
                    "Author": pdf_meta.get("author", "Unknown"),
                    "Extracted Text": text_content
                })

        self.metadata["PDF_Metadata"] = pdf_metadata_list

    # ---------------------- #
    # ✍️ Step 2: Text Extraction & Preprocessing
    # ---------------------- #

    def preprocess_text(self, text):
        """Tokenization, stopword removal, and normalization."""
        doc = nlp(text.lower())  # Convert to lowercase and tokenize
        processed_tokens = [token.text for token in doc if not token.is_stop and not token.is_punct]
        return " ".join(processed_tokens)

    def preprocess_all_pdfs(self):
        """Preprocess text from extracted PDFs."""
        for pdf_data in self.metadata["PDF_Metadata"]:
            pdf_data["Processed Text"] = self.preprocess_text(pdf_data["Extracted Text"])

    # ---------------------- #
    # 🔍 Step 3: Insights Extraction (TF-IDF, NER, Topic Modeling, Sentiment)
    # ---------------------- #

    def extract_insights(self):
        """Apply TF-IDF, Named Entity Recognition, and Topic Modeling to gain insights."""
        self.enriched_metadata["PDF_Metadata"] = []
        texts = [pdf["Processed Text"] for pdf in self.metadata["PDF_Metadata"]]
        
        # TF-IDF Extraction
        vectorizer = TfidfVectorizer(stop_words="english", max_features=10)
        tfidf_matrix = vectorizer.fit_transform(texts)
        keywords = vectorizer.get_feature_names_out()

        # Topic Modeling using LDA
        lda = LatentDirichletAllocation(n_components=3, random_state=42)
        lda.fit(tfidf_matrix)

        # Named Entity Recognition
        for pdf_data in self.metadata["PDF_Metadata"]:
            doc = nlp(pdf_data["Processed Text"])
            named_entities = {ent.label_: ent.text for ent in doc.ents}

            self.enriched_metadata["PDF_Metadata"].append({
                "file_name": pdf_data["File Name"],
                "TF-IDF Keywords": keywords.tolist(),
                "Named Entities": named_entities,
                "Topics Identified": lda.components_.tolist()
            })

    # ---------------------- #
    # 🔗 Step 4: Rule-Based Heuristics for Relationship Discovery
    # ---------------------- #

    def discover_relationships(self):
        """Find relationships using regex and heuristic matching."""
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
            processed_text = pdf_meta.get("Processed Text", "")

            for column in self.metadata.get("DB_Columns", []):
                if column["name"].lower() in processed_text:
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
        """Store extracted metadata in PostgreSQL."""
        conn = self.connect_to_db()
        if not conn:
            return

        cursor = conn.cursor()
        cursor.execute("CREATE TABLE IF NOT EXISTS metadata_without_llm1 (id SERIAL PRIMARY KEY, source TEXT, metadata JSONB);")

        cursor.execute("INSERT INTO metadata_without_llm1 (source, metadata) VALUES (%s, %s);", ("Database", json.dumps(self.metadata)))
        
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
    agent.preprocess_all_pdfs()
    agent.extract_insights()
    agent.discover_relationships()
    agent.save_metadata_to_json()
    agent.store_metadata_in_db()
