import torch
import torch.nn as nn

import sys


class RecommenderModel(nn.Module):
    def __init__(
        self,
        num_users,
        num_books,
        num_authors,
        num_categories,
        num_publishers,
        embedding_dim=50,
    ):
        super(RecommenderModel, self).__init__()

        self.num_users = num_users
        self.num_books = num_books
        self.num_authors = num_authors
        self.num_categories = num_categories
        self.num_publishers = num_publishers

        # Embeddings for user_id and book_id
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.book_embedding = nn.Embedding(num_books, embedding_dim)

        # Embeddings for other categorical features
        self.author_embedding = nn.Embedding(num_authors, embedding_dim)
        self.publisher_embedding = nn.Embedding(num_publishers, embedding_dim)
        # self.category_embedding = nn.Embedding(num_categories, embedding_dim)

        # Final fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(
                embedding_dim * 4 + 9 + 4 + 768, 128  # + 9 for category embeddings
            ),  # Adjust the input size accordingly
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
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

        # Embeddings for user_id and book_id
        user_embed = self.user_embedding(user_id)
        book_embed = self.book_embedding(book_id)

        # Embeddings for other categorical features
        author_embed = self.author_embedding(author_label)
        # category_embed = self.category_embedding(category_label)
        publisher_embed = self.publisher_embedding(publisher_label)

        # Embeddings for other numerical features
        page_count_embed = page_count.unsqueeze(1)
        average_rating_embed = average_rating.unsqueeze(1)
        ratings_count_embed = ratings_count.unsqueeze(1)
        published_year_embed = published_year.unsqueeze(1)

        # Concatenate all features
        x = torch.cat(
            [
                user_embed,
                book_embed,
                author_embed,
                category_label,
                publisher_embed,
                page_count_embed,
                average_rating_embed,
                ratings_count_embed,
                published_year_embed,
                full_text_embeddings,
            ],
            dim=1,
        )

        # Pass through the fully connected layers
        output = self.fc(x)
        return output.squeeze()
