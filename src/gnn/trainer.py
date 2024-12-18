import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
from load_data import load_data
from graph_model import GraphRecommendationModel
import sys
from torch_geometric.data import Data
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from sklearn.model_selection import StratifiedShuffleSplit


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
embedding_dim = 32
num_epochs = 100
learning_rate = 0.0005
validation_split = 0.2
early_stopping_patience = 10000


class EdgeDataset(Dataset):
    def __init__(self, data):
        self.edge_index = data.edge_index
        self.x = data.x
        self.y = data.y

    def __len__(self):
        return self.edge_index.size(1)  # Number of edges

    def __getitem__(self, idx):
        src = self.edge_index[0, idx]
        dst = self.edge_index[1, idx]
        label = self.y[idx]
        return src, dst, label


def split_data(data, val_split=0.1):
    """Split the edges and labels into training and validation sets."""
    user_ids = data.edge_index[0].cpu().numpy()  # Source nodes
    edges = np.arange(data.edge_index.size(1))
    min_interactions = 4

    # count interactions per user
    user_interactions = np.bincount(user_ids, minlength=data.num_nodes)
    rare_users = np.where(user_interactions < min_interactions)[0]
    regular_users = np.where(user_interactions >= min_interactions)[0]

    # seprate rare user edges
    rare_indices = np.isin(user_ids, rare_users)
    rare_val_edges = edges[rare_indices]

    regular_indices = ~rare_indices
    regular_edges = edges[regular_indices]
    regular_user_ids = user_ids[regular_indices]
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=val_split, random_state=42)
    train_indices, val_indices = next(splitter.split(regular_edges, regular_user_ids))
    train_edges = regular_edges[train_indices]
    val_edges = np.concatenate([regular_edges[val_indices], rare_val_edges])
    train_data = Data(
        edge_index=data.edge_index[:, train_edges],
        x=data.x,
        y=data.y[train_edges],
        num_nodes=data.num_nodes,
    )
    val_data = Data(
        edge_index=data.edge_index[:, val_edges],
        x=data.x,
        y=data.y[val_edges],
        num_nodes=data.num_nodes,
    )

    return train_data, val_data


data, num_users, num_books, book_feature_dim, _, _ = load_data()
train_data, val_data = split_data(data, validation_split)
model = GraphRecommendationModel(
    num_users, num_books, book_feature_dim, embedding_dim=32
).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
train_dataset = EdgeDataset(train_data)
val_dataset = EdgeDataset(val_data)

train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)

best_val_loss = float("inf")
patience_counter = 0
best_model_state = None


def validate(model, val_loader, criterion, device, data):
    model.eval()
    total_val_loss = 0

    with torch.no_grad():
        for src_nodes, dst_nodes, labels in val_loader:
            src_nodes = src_nodes.to(device)
            dst_nodes = dst_nodes.to(device)
            labels = labels.to(device)

            predictions = model(
                data.edge_index.to(device), data.x.to(device), src_nodes, dst_nodes
            )
            loss = criterion(predictions, labels)
            total_val_loss += loss.item() * src_nodes.size(0) 

    return total_val_loss / len(val_dataset)

for epoch in range(num_epochs):
    model.train()
    total_train_loss = 0

    for src_nodes, dst_nodes, labels in train_loader:
        src_nodes = src_nodes.to(device)
        dst_nodes = dst_nodes.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        # Since we are working with the full graph, we pass the entire edge_index and x
        predictions = model(
            data.edge_index.to(device), data.x.to(device), src_nodes, dst_nodes
        )
        loss = criterion(predictions, labels)

        loss.backward()
        optimizer.step()
        total_train_loss += loss.item() * src_nodes.size(0) 

    train_loss = total_train_loss / len(train_dataset)
    train_rmse = torch.sqrt(torch.tensor(train_loss))

    val_loss = validate(model, val_loader, criterion, device, data)
    val_rmse = torch.sqrt(torch.tensor(val_loss))

    print(
        f"Epoch {epoch + 1}/{num_epochs}, "
        f"Train Loss: {train_loss:.4f}, Train RMSE: {train_rmse:.4f}, "
        f"Val Loss: {val_loss:.4f}, Val RMSE: {val_rmse:.4f}"
    )

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        best_model_state = model.state_dict().copy()
    else:
        patience_counter += 1

    if patience_counter >= early_stopping_patience:
        print(f"Early stopping triggered after {epoch + 1} epochs")
        break

if best_model_state is not None:
    model.load_state_dict(best_model_state)
torch.save(model.state_dict(), "graph_model.pth")
