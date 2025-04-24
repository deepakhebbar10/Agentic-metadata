#!/usr/bin/env python3
"""
Extended GEMMS‑style ingestion pipeline with:
 1) Plugin‑based Extractor/Parser architecture
 2) DataUnitTemplates driven by config
 3) Multiple DataUnits per file (e.g. Excel sheets)
 4) Semantic annotations via ontology lookup
 5) Full tree structure inference (Algorithm 1) for all formats

Dependencies:
    pip install sqlalchemy psycopg2-binary tika pandas lxml openpyxl pdfminer.six PyPDF2
"""
import io

import os
import json
from abc import ABC, abstractmethod
from collections import deque, defaultdict
from sqlalchemy import (
    Column, Integer, String, Text, JSON, ForeignKey, create_engine
)
from sqlalchemy.orm import declarative_base, relationship, sessionmaker
from tika import parser as tika_parser, detector as tika_detector
import pandas as pd
from lxml import etree
from lxml.etree import Element, SubElement
from pdfminer.high_level import extract_text_to_fp
from pdfminer.layout import LAParams
import openpyxl

# -----------------------------------------------------------------------------
# Load configuration for DataUnitTemplates and semantic mappings
# -----------------------------------------------------------------------------
CONFIG_DIR = os.path.dirname(__file__)
with open(os.path.join(CONFIG_DIR, 'data_unit_templates.json')) as f:
    TEMPLATES = json.load(f)
with open(os.path.join(CONFIG_DIR, 'semantic_mappings.json')) as f:
    SEM_MAP = json.load(f)

# -----------------------------------------------------------------------------
# ORM model definitions
# -----------------------------------------------------------------------------
Base = declarative_base()

class DataFile(Base):
    __tablename__ = 'data_files'
    id         = Column(Integer, primary_key=True)
    path       = Column(Text, unique=True, nullable=False)
    media_type = Column(String, nullable=False)
    size       = Column(Integer)
    modified   = Column(String)
    data_units = relationship("DataUnit", back_populates="data_file")

class DataUnit(Base):
    __tablename__ = 'data_units'
    id             = Column(Integer, primary_key=True)
    file_id        = Column(Integer, ForeignKey('data_files.id'))
    name           = Column(String)
    metadata_props = Column(JSON)
    structure      = Column(JSON)
    annotations    = Column(JSON)   # list of ontology URIs
    data_file      = relationship("DataFile", back_populates="data_units")

# -----------------------------------------------------------------------------
# GEMMS Algorithm 1: Tree Structure Inference
# -----------------------------------------------------------------------------

def infer_tree_structure(root):
    queue, seen = deque([root]), []
    while queue:
        node = queue.popleft()
        seen.append(node)
        queue.extend(list(node))
    summary = {}
    for node in reversed(seen):
        counts = defaultdict(int)
        for child in node:
            counts[child.tag] += 1
        summary[node.tag] = dict(counts)
    return summary

# -----------------------------------------------------------------------------
# Ontology lookup for semantic annotations
# -----------------------------------------------------------------------------

def ontology_lookup(key):
    # simple static mapping from SEM_MAP
    return SEM_MAP.get(key)

# -----------------------------------------------------------------------------
# Base Extractor and registry
# -----------------------------------------------------------------------------
class BaseExtractor(ABC):
    def __init__(self, path, props):
        self.path = path
        self.props = props
    @abstractmethod
    def extract_units(self):
        pass

class ExtractorFactory:
    _registry = {}
    @classmethod
    def register(cls, key):
        def decorator(extcls):
            cls._registry[key] = extcls
            return extcls
        return decorator
    @classmethod
    def get_extractor(cls, path, props):
        ext = os.path.splitext(path)[1].lower().lstrip('.')
        mtype = props.get('Content-Type', '').split(';')[0]
        for key, extcls in cls._registry.items():
            if ext == key or mtype.endswith(key):
                return extcls(path, props)
        return DefaultExtractor(path, props)

# -----------------------------------------------------------------------------
# CSV Extractor using DataUnitTemplates
# -----------------------------------------------------------------------------
@ExtractorFactory.register('csv')
class CSVExtractor(BaseExtractor):
    def extract_units(self):
        df = pd.read_csv(self.path, dtype=str, keep_default_na=False)
        units = []
        tmpl = TEMPLATES.get('csv', [{}])
        for t in tmpl:
            name = t.get('name', os.path.basename(self.path))
            root = Element('table')
            for _, row in df.iterrows():
                r = SubElement(root, 'row')
                for col in df.columns:
                    c = SubElement(r, col)
                    c.text = row[col]
            struct = infer_tree_structure(root)
            anns = [ontology_lookup(col) for col in df.columns if ontology_lookup(col)]
            units.append({'name': name, 'structure': struct, 'annotations': anns})
        return units

# -----------------------------------------------------------------------------
# PDF Extractor via PDFMiner XML tree
# -----------------------------------------------------------------------------
@ExtractorFactory.register('pdf')
class PDFExtractor(BaseExtractor):
    def extract_units(self):
        buf = io.BytesIO()
        with open(self.path, 'rb') as inf:
            extract_text_to_fp(inf, buf, output_type='xml', laparams=LAParams())
        buf.seek(0)
        root = etree.fromstring(buf.read(), parser=etree.XMLParser(recover=True))
        struct = infer_tree_structure(root)
        anns = [ontology_lookup(k) for k in self.props.keys() if ontology_lookup(k)]
        return [{'name': os.path.basename(self.path), 'structure': struct, 'annotations': anns}]

# -----------------------------------------------------------------------------
# XML Extractor
# -----------------------------------------------------------------------------
@ExtractorFactory.register('xml')
class XMLExtractor(BaseExtractor):
    def extract_units(self):
        root = etree.parse(self.path).getroot()
        struct = infer_tree_structure(root)
        anns = [ontology_lookup(elem) for elem in self.props.keys() if ontology_lookup(elem)]
        return [{'name': os.path.basename(self.path), 'structure': struct, 'annotations': anns}]

# -----------------------------------------------------------------------------
# Default Extractor
# -----------------------------------------------------------------------------
class DefaultExtractor(BaseExtractor):
    def extract_units(self):
        # fallback: minimal structure
        root = Element('file')
        struct = infer_tree_structure(root)
        anns = [ontology_lookup(k) for k in self.props.keys() if ontology_lookup(k)]
        return [{'name': os.path.basename(self.path), 'structure': struct, 'annotations': anns}]

# -----------------------------------------------------------------------------
# MetadataManager orchestrates ingestion
# -----------------------------------------------------------------------------
class MetadataManager:
    def __init__(self, session): self.session = session
    def ingest_directory(self, root_dir):
        for dp, _, fns in os.walk(root_dir):
            for fn in fns:
                path = os.path.join(dp, fn)
                try: self.process_file(path)
                except Exception as e: print(f"[ERROR] {fn}: {e}")
        self.session.commit()
    def process_file(self, path):
        st = os.stat(path)
        raw = tika_parser.from_file(path)
        props = raw.get('metadata', {})
        df = DataFile(path=path, media_type=tika_detector.from_file(path),
                      size=st.st_size, modified=str(st.st_mtime))
        self.session.add(df); self.session.flush()
        extractor = ExtractorFactory.get_extractor(path, props)
        for unit in extractor.extract_units():
            du = DataUnit(file_id=df.id, name=unit['name'],
                          metadata_props=props,
                          structure=unit['structure'],
                          annotations=unit['annotations'])
            self.session.add(du)

# -----------------------------------------------------------------------------
# Main entrypoint
# -----------------------------------------------------------------------------
if __name__ == '__main__':
    DB_URL = "postgresql+psycopg2://postgres:deepak@localhost:5432/metadata"
    eng = create_engine(DB_URL); Base.metadata.create_all(eng)
    sess = sessionmaker(bind=eng)()
    DATA_DIR = r'C:\Users\deepa\Desktop\Minor Project Meta Data Agent\data\all'
    MetadataManager(sess).ingest_directory(DATA_DIR)
    print('✔ Ingestion complete.')
