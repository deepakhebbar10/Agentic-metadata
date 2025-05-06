# AutoGen Metadata Pipeline – **quick‑start**

This repo wraps your three original Python scripts in **AutoGen conversational agents** and
glues them together in a round‑robin team so they run fully unattended.

```
work/
├── agents/
│   ├── metadata_discovery_autogen.py
│   ├── metadata_enrichment_autogen.py
│   └── relationship_autogen.py
├── data/
│   └── pdfs/              ← put your invoice PDFs here
├── config.py              ← edit DB credentials here
├── run_pipeline.py        ← start the show
├── requirements.txt
└── README.md
```

## 1  Set up

```bash
# Clone / copy the folder, then inside it:
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

✨ **Choose your LLM**  
This template defaults to `gpt-4o`.  
If you prefer Ollama (e.g. Mistral‑7B), change `OpenAIChatCompletionClient`
to `OllamaChatCompletionClient` in the three agent files.

## 2  Prepare your data

1. **Postgres**  
   Ensure a table **`product_reviews`** exists in the database configured in `config.py`.
2. **PDF invoices**  
   Drop all your invoice PDFs in `data/pdfs/`.
   The discovery agent reads only the first three pages of each file.

## 3  Run the pipeline

```bash
python run_pipeline.py
```

Console output shows each tool call (✅/❌).  
On success you will see three JSON artefacts in the project root:

* `discovery_output.json`
* `enriched_output.json`
* `relationships_output.json`

Inspect them or feed them downstream into Neo4j for visualisation.

## 4  Troubleshooting

* _`psycopg2.OperationalError`_ → wrong DB credentials in `config.py`.
* _`ModuleNotFoundError: fitz`_ → `pip install PyMuPDF` inside the active venv.
* _LLM quota errors_ → Reduce PDF pages scanned or switch to a local model via Ollama.

Happy hacking! – *Last built: 2025-05-01*
