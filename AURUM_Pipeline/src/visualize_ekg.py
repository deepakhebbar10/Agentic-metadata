import networkx as nx
import matplotlib.pyplot as plt
import argparse

def visualize_all_nodes(graph_path, save_as=None):
    G = nx.read_graphml(graph_path)

    print(f"📊 Nodes: {len(G.nodes)} | Edges: {len(G.edges)}")

    # Set up plot
    plt.figure(figsize=(20, 20))
    pos = nx.spring_layout(G, k=0.4, seed=42)  # Better spacing

    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_color='skyblue', node_size=800)

    # Draw edges (with thickness and color)
    nx.draw_networkx_edges(G, pos, alpha=0.6, edge_color='gray', width=2)

    # Draw node labels (shortened if too long)
    labels = {node: node.split(":")[-1][:30] for node in G.nodes}
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=8, font_weight='bold')

    plt.title("📘 Enterprise Knowledge Graph", fontsize=16)
    plt.axis("off")

    # Save or show
    if save_as:
        plt.tight_layout()
        plt.savefig(save_as, dpi=300)
        print(f"🖼️ Graph saved to {save_as}")
    else:
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph", required=True, help="Path to the GraphML file")
    parser.add_argument("--output", help="Optional: Save to image file (e.g., ekg.png)")
    args = parser.parse_args()

    visualize_all_nodes(args.graph, args.output)


