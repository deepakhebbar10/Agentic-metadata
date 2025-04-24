import os
import json
import networkx as nx
from datasketch import MinHash, MinHashLSH
import argparse
import numpy as np
def load_profiles(profile_dir):
    profiles = {}
    for file in os.listdir(profile_dir):
        if file.endswith(".json"):
            with open(os.path.join(profile_dir, file)) as f:
                profiles[file] = json.load(f)
    return profiles


def build_graph(profiles):
           lsh=MinHashLSH(threshold=0.125,num_perm=128)
           mh_dict={}
           G=nx.Graph()
           for f,cols in profiles.items():
                  for col,meta in cols.items():
                         name=f"{f}:{col}"
                         mh=MinHash(num_perm=128)
                         mh.hashvalues=np.array(meta["minhash"],dtype=np.uint64)
                         mh_dict[name]=mh
                         lsh.insert(name,mh)
                         G.add_node(name,unique_ratio=meta["unique_ratio"])
           for name,mh in mh_dict.items():
                  for similar in lsh.query(mh):
                         if name!=similar and not G.has_edge(name,similar):
                                G.add_edge(name,similar)

           print(f"Nodes: {len(G.nodes)}, Edges: {len(G.edges)}")
           return G

def save_graph(graph, output_path):
    nx.write_graphml(graph, output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile_dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    profiles = load_profiles(args.profile_dir)
    graph = build_graph(profiles)
	
    save_graph(graph, args.output)
