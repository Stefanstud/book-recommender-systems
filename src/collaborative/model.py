import torch
import torch.nn as nn


class CollaborativeFilteringModel(nn.Module):
    def __init__(self, num_users, num_books, embedding_dim=50):
        super(CollaborativeFilteringModel, self).__init__()
        self.user_embedding = nn.Embedding(
            num_users + 1, embedding_dim
        )  # +1 for unknown users
        self.book_embedding = nn.Embedding(
            num_books + 1, embedding_dim
        )  # +1 for unknown books
        self.fc = nn.Linear(embedding_dim, 1)

    def forward(self, user_id, book_id):
        user_emb = self.user_embedding(user_id)
        book_emb = self.book_embedding(book_id)
        interaction = user_emb * book_emb
        output = self.fc(interaction).squeeze(1)
        return output
