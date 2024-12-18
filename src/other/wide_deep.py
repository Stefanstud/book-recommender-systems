import torch
import torch.nn as nn


class WideAndDeepModel(nn.Module):
    def __init__(
        self,
        num_users,
        num_books,
        num_authors,
        num_categories,
        num_publishers,
        embedding_dim=32,
    ):
        super(WideAndDeepModel, self).__init__()

        # Wide component: Linear layer for input features
        self.wide = nn.Linear(4, 1)

        # Deep component: Embeddings and MLP
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.book_embedding = nn.Embedding(num_books, embedding_dim)
        self.author_embedding = nn.Embedding(num_authors, embedding_dim)
        self.category_embedding = nn.Embedding(num_categories, embedding_dim)
        self.publisher_embedding = nn.Embedding(num_publishers, embedding_dim)

        # MLP layers
        self.deep = nn.Sequential(
            nn.Linear(
                5 * embedding_dim + 4 + 768, 128
            ),  # 5 embeddings + 5 numerical features + 768 full_text
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
        )

    def forward(
        self,
        user_id,
        book_id,
        author_label,
        category_label,
        publisher_label,
        page_count,
        average_rating,
        ratings_count,
        published_year,
        full_text_embeddings,
    ):

        # Wide component
        wide_input = torch.stack(
            [page_count, average_rating, ratings_count, published_year], dim=1
        )
        wide_output = self.wide(wide_input)

        # Deep component
        user_emb = self.user_embedding(user_id)
        book_emb = self.book_embedding(book_id)
        author_emb = self.author_embedding(author_label)
        category_emb = self.category_embedding(category_label)  # 32 9 32
        category_emb = category_emb.mean(dim=1)
        publisher_emb = self.publisher_embedding(publisher_label)

        # print("user_emb:", user_emb.shape)
        # print("book_emb:", book_emb.shape)
        # print("author_emb:", author_emb.shape)
        # print("category_emb:", category_emb.shape)
        # print("publisher_emb:", publisher_emb.shape)
        # print("page_count:", page_count.unsqueeze(1).shape)
        # print("average_rating:", average_rating.unsqueeze(1).shape)
        # print("ratings_count:", ratings_count.unsqueeze(1).shape)
        # print("published_year:", published_year.unsqueeze(1).shape)
        # print("full_text_embeddings:", full_text_embeddings.squeeze().shape)

        # Concatenate embeddings and numerical features
        deep_input = torch.cat(
            [
                user_emb,
                book_emb,
                author_emb,
                category_emb,
                publisher_emb,
                page_count.unsqueeze(1),
                average_rating.unsqueeze(1),
                ratings_count.unsqueeze(1),
                published_year.unsqueeze(1),
                full_text_embeddings,
            ],
            dim=1,
        )
        deep_output = self.deep(deep_input)

        # Combine wide and deep components
        output = wide_output + deep_output
        return output.squeeze()
