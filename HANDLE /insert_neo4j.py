from py2neo import Graph, Node, Relationship
import os, json

graph = Graph("bolt://localhost:7687", auth=("neo4j", "password"))
metadata_dir = "./metadata_json"

for file in os.listdir(metadata_dir):
    with open(os.path.join(metadata_dir, file)) as f:
        data = json.load(f)

    data_node = Node("Data", id=data['file_name'], path=data['path'])
    graph.merge(data_node, "Data", "id")

    meta_node = Node("Metadata", type="Descriptive")
    graph.create(meta_node)
    graph.create(Relationship(data_node, "HAS_METADATA", meta_node))

    for key, value in data.items():
        if key not in ['file_name', 'path']:
            prop_node = Node("Property", key=key, value=str(value))
            graph.create(prop_node)
            graph.create(Relationship(meta_node, "HAS_PROPERTY", prop_node))
