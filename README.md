Metadata Management System for Data Sources
This repository contains the implementation of various agents designed to extract, enrich, and discover relationships in metadata from diverse data sources like PostgreSQL databases and PDF files.
It aims to provide a flexible framework for metadata management that enables the extraction of useful insights, enrichment with domain-specific information, and the discovery of relationships 
between various metadata elements.

Agents Overview
The repository consists of the following agents:

MetadataDiscoveryAgent: Extracts metadata from PostgreSQL databases and PDF files, including details about database schema and raw data.

MetadataEnrichmentAgent: Enhances the extracted metadata by applying Named Entity Recognition (NER) and rule-based enrichment strategies, and generates semantic annotations for metadata fields.

RelationshipDiscoveryAgent: Identifies and establishes relationships between different metadata fields using heuristic rules and insights derived from the extracted metadata.

Prerequisites
Before using these agents, ensure that you have the following installed on your machine:

Python 3.x

PostgreSQL database (with access to the product_reviews table or any similar dataset)

Required Python libraries (listed below)

Installation
Clone this repository to your local machine:

bash
Copy
git clone https://github.com/yourusername/metadata-management-system.git
cd metadata-management-system
Install the required dependencies:

bash
Copy
pip install -r requirements.txt
Dependencies
psycopg2: PostgreSQL database adapter for Python

PyMuPDF: PDF processing (used for extracting metadata and text from PDFs)

pdfplumber: PDF text extraction library

spaCy: NLP model for Named Entity Recognition (NER)

Tesseract: OCR library (for extracting text from scanned PDFs)

ollama: For interacting with the Mistral LLM to analyze text and metadata

sklearn: For extracting features like TF-IDF and performing Latent Dirichlet Allocation (LDA) topic modeling

json: To handle JSON file storage

Configuration
Before running the agents, ensure that your PostgreSQL database is set up and running, and update the DB_CONFIG dictionary in the code to match your PostgreSQL connection details:

python
Copy
DB_CONFIG = {
    "dbname": "metadata",
    "user": "postgres",
    "password": "yourpassword",
    "host": "localhost",
    "port": "5432"
}
For PDF extraction, place your PDF files in the folder specified in the pdf_dir parameter of the agents.

Agent 1: MetadataDiscoveryAgent
Purpose: Extracts metadata from PostgreSQL database tables and PDFs.

How to Use:

Configure your database connection in the DB_CONFIG dictionary.

Specify the path to the folder containing the PDFs.

Run the agent as follows:

bash
Copy
python MetadataDiscoveryAgent.py
What it does:

Extracts metadata such as column names, data types, and row counts from a PostgreSQL database (e.g., product_reviews table).

Extracts metadata from PDFs and uses Mistral LLM for insights generation from the content.

Output:

Metadata is saved in metadata.json and stored in a PostgreSQL table.

Agent 2: MetadataEnrichmentAgent
Purpose: Enriches the extracted metadata using Named Entity Recognition (NER) and rule-based enrichment strategies.

How to Use:

Run the MetadataDiscoveryAgent first to extract metadata.

Then run the MetadataEnrichmentAgent:

bash
Copy
python MetadataEnrichmentAgent.py
What it does:

Applies NER using spaCy to identify named entities (e.g., product names, dates, quantities) in the extracted PDF metadata.

Performs rule-based enrichment of database columns (e.g., marking columns as "primary key", "date", "price", etc.).

Output:

Enriched metadata is saved in metadata_enriched.json and stored in a separate table in PostgreSQL.

Agent 3: RelationshipDiscoveryAgent
Purpose: Discovers relationships between metadata fields using heuristic rules and insights derived from the enriched metadata.

How to Use:

Run both MetadataDiscoveryAgent and MetadataEnrichmentAgent to extract and enrich metadata.

Run the RelationshipDiscoveryAgent:

bash
Copy
python RelationshipDiscoveryAgent.py
What it does:

Finds relationships between different metadata fields using heuristic rules (e.g., linking price with product_name, review_rating with product_id, etc.).

Links PDF metadata to database columns based on matching keywords.

Uses Mistral LLM to analyze and generate relationship insights.

Output:

Relationships between metadata fields are stored in metadata_relationships.json and a PostgreSQL table.

Example Workflow
Extract Metadata: Use MetadataDiscoveryAgent to extract metadata from the PostgreSQL database and PDFs.

Enrich Metadata: Use MetadataEnrichmentAgent to enhance the extracted metadata with semantic annotations and named entities.

Discover Relationships: Use RelationshipDiscoveryAgent to find relationships between metadata fields and generate insights using Mistral LLM.

Data Formats and Storage
The extracted, enriched, and relational metadata are stored in:

JSON Files: metadata.json, metadata_enriched.json, and metadata_relationships.json.

PostgreSQL: Each agent stores data in a table with the format metadata_table_name (e.g., metadata_without_llm, metadata_enriched_without_llm, etc.).

Contributions
Contributions are welcome! If you have any ideas for improvements or additional features, feel free to fork this repository, create a pull request, or open an issue.

License
This project is licensed under the MIT License - see the LICENSE file for details.

Final Note
This metadata management system provides a comprehensive approach to extracting, enriching, and discovering relationships between data from various sources. It integrates NLP, LLM, and database metadata extraction to create an adaptable and extensible framework for working with data lakes.
