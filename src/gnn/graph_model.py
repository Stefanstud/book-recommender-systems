import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv
import torch.nn.functional as F

class GraphRecommendationModel(nn.Module):
    def __init__(self, num_users, num_books, book_feature_dim, embedding_dim, dropout=0.2):
        super(GraphRecommendationModel, self).__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.book_embedding = nn.Embedding(num_books, embedding_dim)
        # self.book_features = nn.Linear(book_feature_dim, embedding_dim)

        # GNN Layers
        self.conv1 = GCNConv(embedding_dim, embedding_dim)
        self.conv2 = GCNConv(embedding_dim, embedding_dim)
        self.bn1 = nn.BatchNorm1d(embedding_dim)
        self.bn2 = nn.BatchNorm1d(embedding_dim)
        self.dropout = nn.Dropout(p=dropout)

        # Initialize embeddings
        nn.init.normal_(self.user_embedding.weight, std=0.01)
        nn.init.normal_(self.book_embedding.weight, std=0.01)
        # nn.init.xavier_uniform_(self.book_features.weight)

    def forward(self, edge_index, x, src_nodes, dst_nodes):
        num_users = self.user_embedding.num_embeddings
        user_emb = self.user_embedding.weight  # Shape: [num_users, emb_dim]
        book_emb = self.book_embedding.weight

        #book_emb = self.book_features(x[num_users:])  # Shape: [num_books, emb_dim]
        x = torch.cat([user_emb, book_emb], dim=0)  # Shape: [num_users + num_books, emb_dim]
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout(x)

        user_nodes = x[src_nodes]  # Shape: [batch_size, emb_dim]
        book_nodes = x[dst_nodes]  # Shape: [batch_size, emb_dim]

        # dot product 
        ratings = (user_nodes * book_nodes).sum(dim=1)  # Shape: [batch_size]
        return ratings
