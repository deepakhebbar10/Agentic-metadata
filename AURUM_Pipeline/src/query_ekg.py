import argparse
import networkx as nx

def query_graph(graph_path, query_column):
    G = nx.read_graphml(graph_path)
    results = [n for n in G.nodes if query_column.lower() in n.lower()]
    for r in results:
        print(f"Match: {r}")
        print("Neighbors:")
        for nbr in G.neighbors(r):
            print(f"  - {nbr}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--query_column", required=True)
    parser.add_argument("--graph", required=True)
    args = parser.parse_args()
    query_graph(args.graph, args.query_column)
