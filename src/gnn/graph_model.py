import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, SAGEConv, GATv2Conv


class GraphRecommendationModel(nn.Module):
    def __init__(self, num_users, num_books, book_feature_dim, embedding_dim):
        super(GraphRecommendationModel, self).__init__()
        # User and book embeddings
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.book_embedding = nn.Embedding(num_books, embedding_dim)
        self.book_features = nn.Linear(book_feature_dim, embedding_dim)

        # GNN
        self.conv1 = GCNConv(embedding_dim, embedding_dim)
        self.conv2 = GCNConv(embedding_dim, embedding_dim)

        # Edge prediction (for rating)
        self.edge_predictor = nn.Sequential(
            nn.Linear(2 * embedding_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),  # Predicting a single rating value
        )

    def forward(self, edge_index, x, src_nodes, dst_nodes):
        # Split the input features into user and book embeddings
        num_users = self.user_embedding.num_embeddings
        user_emb = self.user_embedding(torch.arange(num_users, device=x.device))
        book_emb = self.book_features(x[num_users:])

        # Combine user and book embeddings
        x = torch.cat([user_emb, book_emb], dim=0)

        # GNN
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)

        # Predict ratings for edges
        user_nodes = x[src_nodes]
        book_nodes = x[dst_nodes]
        edge_features = torch.cat([user_nodes, book_nodes], dim=1)
        ratings = self.edge_predictor(edge_features).squeeze()
        return ratings
