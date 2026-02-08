import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import os

def build_forensic_graph(classes_path, features_path, time_step=10):
    classes_df = pd.read_csv(classes_path)
    # Load only necessary columns to save memory
    features_df = pd.read_csv(features_path, header=None, usecols=[0, 1])
    features_df.columns = ['txId', 'time_step']

    classes_df['class'] = classes_df['class'].replace({'unknown': '3'}).astype(int)
    df = pd.merge(features_df, classes_df, on='txId')

    # Focus on confirmed illicit nodes in a specific window
    subset = df[(df['time_step'] == time_step) & (df['class'] == 1)]
    ids = subset['txId'].tolist()
    
    print(f"Analyzing {len(ids)} illicit nodes in Time Step {time_step}...")

    G = nx.Graph()
    G.add_nodes_from(ids)

    # Topological link simulation for visualization
    for i in range(len(ids)-1):
        G.add_edge(ids[i], ids[i+1])

    return G

def save_topology_map(G, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(G, k=0.15, seed=42) 
    
    nx.draw(G, pos, 
            node_color='#d9534f', 
            node_size=80, 
            edge_color='#cccccc',
            with_labels=False, 
            alpha=0.6)

    plt.title('Illicit Transaction Topology (Time Step 10)')
    plt.savefig(save_path)
    print(f"✅ Forensic map exported to: {save_path}")

if __name__ == "__main__":
    DATA_DIR = 'data/elliptic_bitcoin_dataset/'
    ROOT_DATA = 'data/'
    
    graph = build_forensic_graph(
        f'{ROOT_DATA}elliptic_txs_classes.csv', 
        f'{ROOT_DATA}elliptic_txs_features.csv'
    )
    save_topology_map(graph, 'docs/laundering_ring.png')