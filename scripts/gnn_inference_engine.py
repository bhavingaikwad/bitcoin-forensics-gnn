import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from sklearn.metrics import classification_report

class GCN(torch.nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.conv1 = GCNConv(input_dim, 32)
        self.conv2 = GCNConv(32, 16)
        self.conv3 = GCNConv(16, 2)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.3, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        x = self.conv3(x, edge_index)
        return F.log_softmax(x, dim=1)

def run_gnn_inference(root_data, edgelist_path):
    # 1. Load Data
    classes = pd.read_csv(f'{root_data}elliptic_txs_classes.csv')
    features = pd.read_csv(f'{root_data}elliptic_txs_features.csv', header=None)
    edges = pd.read_csv(edgelist_path)

    classes['class'] = classes['class'].replace({'unknown': '3'}).astype(int)
    labeled_df = pd.merge(features, classes, left_on=0, right_on='txId')
    labeled_df = labeled_df[labeled_df['class'].isin([1, 2])].reset_index(drop=True)
    
    # 2. Build Mapping & Edges
    id_map = {old_id: new_idx for new_idx, old_id in enumerate(labeled_df[0].values)}
    valid_edges = edges[edges['txId1'].isin(id_map) & edges['txId2'].isin(id_map)]
    
    edge_index = torch.tensor([
        [id_map[id1] for id1 in valid_edges['txId1']],
        [id_map[id2] for id2 in valid_edges['txId2']]
    ], dtype=torch.long)

    # 3. Prepare Tensors
    x = torch.tensor(labeled_df.drop([0, 1, 'txId', 'class'], axis=1).values, dtype=torch.float)
    y = torch.tensor(labeled_df['class'].apply(lambda x: 1 if x == 1 else 0).values, dtype=torch.long)
    
    train_mask = torch.tensor(labeled_df[1].values <= 34)
    test_mask = torch.tensor(labeled_df[1].values > 34)

    data = Data(x=x, edge_index=edge_index, y=y)
    
    # 4. Weighted Training
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GCN(x.shape[1]).to(device)
    data = data.to(device)
    weights = torch.tensor([1.0, 5.0]).to(device) # Boost Illicit weight
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    print(f"🚀 Training Forensic GNN on {device}...")
    model.train()
    for epoch in range(101):
        optimizer.zero_grad()
        out = model(data)
        loss = F.nll_loss(out[train_mask], data.y[train_mask], weight=weights)
        loss.backward()
        optimizer.step()
        if epoch % 25 == 0: print(f"Epoch {epoch} | Loss: {loss.item():.4f}")

    return data.y[test_mask].cpu(), model(data).argmax(dim=1)[test_mask].cpu()

if __name__ == "__main__":
    y_true, y_pred = run_gnn_inference('data/', 'data/elliptic_bitcoin_dataset/elliptic_txs_edgelist.csv')
    print(f"\n{'='*40}\nGNN FINAL INFERENCE REPORT\n{'='*40}")
    print(classification_report(y_true, y_pred, target_names=['Licit', 'Illicit']))