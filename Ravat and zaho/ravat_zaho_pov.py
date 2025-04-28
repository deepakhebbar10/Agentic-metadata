#!/usr/bin/env python3
"""
Complete MEDAL-style metadata pipeline including:
- Discovery (CSV, PDF, PostgreSQL table)
- Enrichment (semantic types)
- Structural relationship discovery (shares-attribute)
- Provenance discovery (MD5-based lineage)
- Dual storage in PostgreSQL and Neo4j
"""

import os
import glob
import json
import hashlib
import pandas as pd
from datetime import datetime
from itertools import combinations

from PyPDF2 import PdfReader
from sqlalchemy import (
    create_engine, Column, Integer, String, Text, JSON, inspect, text
)
from sqlalchemy.orm import declarative_base, sessionmaker
from neo4j import GraphDatabase

# --- Configuration ---
REL_DB_URI    = os.getenv(
    "REL_DB_URI",
    "postgresql+psycopg2://postgres:deepak@localhost:5432/metadata"
)
NEO4J_URI     = os.getenv("NEO4J_URI",  "bolt://localhost:7687")
NEO4J_USER    = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASS    = os.getenv("NEO4J_PASS", "deepakhebbar")
PRODUCT_TABLE = os.getenv("PRODUCT_TABLE", "product_reviews")
BASE_FOLDER   = os.getenv("BASE_FOLDER", "C:/Users/deepa/Desktop/Minor Project Meta Data Agent/data/all2")


# --- SQLAlchemy setup ---
Base = declarative_base()
engine = create_engine(REL_DB_URI, echo=False)
Session = sessionmaker(bind=engine)

class Dataset(Base):
    __tablename__ = 'datasets'
    id    = Column(Integer, primary_key=True)
    name  = Column(String, unique=True, nullable=False)
    type  = Column(String)       # 'csv', 'pdf', 'table'
    path  = Column(Text, nullable=True)
    meta  = Column('metadata', JSON)

class Attribute(Base):
    __tablename__ = 'attributes'
    id         = Column(Integer, primary_key=True)
    dataset_id = Column(Integer, nullable=False)
    name       = Column(String, nullable=False)
    dtype      = Column(String)
    stats      = Column(JSON)

Base.metadata.create_all(engine)

# --- Neo4j setup ---
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))

# --- Utility functions ---
def compute_md5(path, block_size=65536):
    md5 = hashlib.md5()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(block_size), b''):
            md5.update(chunk)
    return md5.hexdigest()

def get_mod_time(path):
    ts = os.path.getmtime(path)
    return datetime.fromtimestamp(ts).isoformat()

# --- Metadata Discovery ---
class MetadataDiscoveryAgent:
    @staticmethod
    def discover_csv(path):
        df = pd.read_csv(path)
        meta = {
            "num_rows": df.shape[0],
            "num_columns": df.shape[1],
            "columns": list(df.columns),
            "md5": compute_md5(path),
            "last_modified": get_mod_time(path)
        }
        attrs = [
            {"name": col,
             "dtype": str(df[col].dtype),
             "stats": {
                 "unique": int(df[col].nunique()),
                 "nulls": int(df[col].isna().sum())
             }}
            for col in df.columns
        ]
        return meta, attrs

    @staticmethod
    def discover_pdf(path):
        reader = PdfReader(path)
        info = reader.metadata or {}
        meta = {
            "num_pages": len(reader.pages),
            "title": info.title,
            "author": info.author,
            "md5": compute_md5(path),
            "last_modified": get_mod_time(path)
        }
        attrs = [
            {"name": f"page_{i+1}", "dtype": "page", "stats": {}}
            for i in range(len(reader.pages))
        ]
        return meta, attrs

    @staticmethod
    def discover_db_table(table_name, engine):
        insp = inspect(engine)
        cols = insp.get_columns(table_name)
        with engine.connect() as conn:
            row_count = conn.execute(text(f"SELECT COUNT(*) FROM {table_name}")).scalar()
        meta = {
            "table_rows": int(row_count),
            "md5": None,
            "last_modified": None
        }
        attrs = []
        with engine.connect() as conn:
            for col in cols:
                name = col["name"]
                uq = conn.execute(text(f"SELECT COUNT(DISTINCT {name}) FROM {table_name}")).scalar()
                nl = conn.execute(text(f"SELECT COUNT(*) - COUNT({name}) FROM {table_name}")).scalar()
                attrs.append({
                    "name": name,
                    "dtype": str(col["type"]),
                    "stats": {"unique": int(uq), "nulls": int(nl)}
                })
        return meta, attrs

# --- Metadata Enrichment ---
class MetadataEnrichmentAgent:
    @staticmethod
    def enrich(meta, attrs):
        sem_summary = {}
        for a in attrs:
            name = a["name"]
            dtype = a["dtype"]
            if dtype.startswith(("int", "float")):
                sem = "Numerical"
            elif "date" in name.lower():
                sem = "Date"
            else:
                sem = "Categorical"
            a["stats"]["semantic_type"] = sem
            sem_summary[sem] = sem_summary.get(sem, 0) + 1
        enriched_meta = dict(meta)
        enriched_meta["semantic_summary"] = sem_summary
        return enriched_meta, attrs

# --- Structural Relationship Discovery ---
class RelationshipDiscoveryAgent:
    @staticmethod
    def discover_shared_attributes():
        session_sql = Session()
        sql = text("""
            SELECT name AS attr_name,
                   array_agg(DISTINCT dataset_id) AS ds_ids
            FROM attributes
            GROUP BY name
            HAVING COUNT(DISTINCT dataset_id) > 1
        """)
        rows = session_sql.execute(sql).all()
        ds_map = {ds.id: ds.name for ds in session_sql.query(Dataset).all()}
        session_sql.close()

        with driver.session() as session_neo4j:
            for attr_name, ds_ids in rows:
                for id1, id2 in combinations(ds_ids, 2):
                    name1 = ds_map.get(id1)
                    name2 = ds_map.get(id2)
                    if name1 and name2:
                        session_neo4j.run(
                            "MATCH (a:Dataset {name:$n1}), (b:Dataset {name:$n2}) "
                            "MERGE (a)-[:SHARES_ATTRIBUTE {attribute:$attr}]->(b)",
                            n1=name1, n2=name2, attr=attr_name
                        )

# --- Provenance Discovery ---
class ProvenanceAgent:
    @staticmethod
    def discover_file_copy_lineage():
        session_sql = Session()
        all_ds = session_sql.query(Dataset).all()
        session_sql.close()

        groups = {}
        for ds in all_ds:
            if isinstance(ds.meta, dict):
                md5 = ds.meta.get('md5')
                lm  = ds.meta.get('last_modified')
                if md5 and lm:
                    groups.setdefault(md5, []).append((ds.name, lm))

        print("Provenance groups:", {k: len(v) for k, v in groups.items() if len(v) > 1})

        with driver.session() as session_neo4j:
            for md5, items in groups.items():
                if len(items) < 2:
                    continue
                sorted_items = sorted(
                    [(n, datetime.fromisoformat(lm)) for n, lm in items],
                    key=lambda x: x[1]
                )
                for (older, _), (newer, _) in zip(sorted_items, sorted_items[1:]):
                    session_neo4j.run(
                        "MATCH (o:Dataset {name:$older}), (n:Dataset {name:$newer}) "
                        "MERGE (n)-[:DERIVED_FROM {md5:$md5}]->(o)",
                        older=older, newer=newer, md5=md5
                    )

# --- Storage Helpers ---
def store_relational(name, ds_type, path, meta, attrs):
    session = Session()
    ds = session.query(Dataset).filter_by(name=name).first()
    if ds:
        ds.type, ds.path, ds.meta = ds_type, path, meta
        session.commit()
        session.query(Attribute).filter_by(dataset_id=ds.id).delete()
        session.commit()
    else:
        ds = Dataset(name=name, type=ds_type, path=path, meta=meta)
        session.add(ds)
        session.commit()
    for a in attrs:
        session.add(Attribute(dataset_id=ds.id, name=a['name'], 
                              dtype=a['dtype'], stats=a['stats']))
    session.commit()
    session.close()

def store_graph(name, ds_type, path, meta, attrs):
    with driver.session() as session_neo4j:
        session_neo4j.run(
            "MERGE (d:Dataset {name:$name}) SET d.type=$type, d.path=$path, d.meta=$meta",
            name=name, type=ds_type, path=path, meta=json.dumps(meta)
        )
        for a in attrs:
            session_neo4j.run(
                "MATCH (d:Dataset {name:$name}) MERGE (c:Attribute {name:$col}) "
                "SET c.dtype=$dtype, c.stats=$stats MERGE (d)-[:HAS_ATTRIBUTE]->(c)",
                name=name, col=a['name'], 
                dtype=a['dtype'], stats=json.dumps(a['stats'])
            )

# --- Main Pipeline ---
if __name__ == "__main__":
    inputs = []
    for path in glob.glob(os.path.join(BASE_FOLDER, "**", "*.csv"), recursive=True):
        inputs.append((os.path.splitext(os.path.basename(path))[0], "csv", path))
    for path in glob.glob(os.path.join(BASE_FOLDER, "**", "*.pdf"), recursive=True):
        inputs.append((os.path.splitext(os.path.basename(path))[0], "pdf", path))

    for name, ds_type, path in inputs:
        if ds_type == "csv":
            meta, attrs = MetadataDiscoveryAgent.discover_csv(path)
        else:
            meta, attrs = MetadataDiscoveryAgent.discover_pdf(path)
        meta, attrs = MetadataEnrichmentAgent.enrich(meta, attrs)
        store_relational(name, ds_type, path, meta, attrs)
        store_graph(name, ds_type, path, meta, attrs)

    meta, attrs = MetadataDiscoveryAgent.discover_db_table(PRODUCT_TABLE, engine)
    meta, attrs = MetadataEnrichmentAgent.enrich(meta, attrs)
    store_relational(PRODUCT_TABLE, "table", PRODUCT_TABLE, meta, attrs)
    store_graph(PRODUCT_TABLE, "table", PRODUCT_TABLE, meta, attrs)

    RelationshipDiscoveryAgent.discover_shared_attributes()
    ProvenanceAgent.discover_file_copy_lineage()

    print("Pipeline complete: discovery, enrichment, relationships, provenance.")
