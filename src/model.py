import torch
import torch.nn as nn

import sys


class RecommenderModel(nn.Module):
    def __init__(
        self,
        num_users,
        num_books,
        num_authors,
        cat_dim,
        num_publishers,
        num_langs,
        embedding_dim=128,
    ):
        super(RecommenderModel, self).__init__()

        self.num_users = num_users
        self.num_books = num_books
        self.num_authors = num_authors
        self.cat_dim = cat_dim
        self.num_publishers = num_publishers

        # Embeddings for user_id and book_id
        self.user_embedding = nn.Embedding(
            num_users, embedding_dim
        )  # make emb dim higher for these ones
        self.book_embedding = nn.Embedding(num_books, embedding_dim)

        # -----
        self.language_embedding = nn.Embedding(num_langs, 8)
        self.maturity_rating_embedding = nn.Embedding(2, 1)

        # Embeddings for other categorical features
        self.author_proj = nn.Linear(num_authors, embedding_dim // 2)  # TODO: define
        self.publisher_proj = nn.Linear(num_publishers, embedding_dim // 2)
        self.category_proj = nn.Linear(cat_dim, embedding_dim // 2)

        input = 3.5 * embedding_dim + 8 + 1 + 768 + 2
        # 3.5 * embedding_dim + 8 + 1 + 768 (full_text)
        # Final fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(int(input), 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
        )

    def forward(
        self,
        user_id,
        book_id,
        authors,
        categories,
        publisher,
        page_count,
        published_year,
        language,
        maturity_rating,
        full_text_embeddings,
    ):

        # Embeddings for user_id and book_id
        user_embed = self.user_embedding(user_id)
        book_embed = self.book_embedding(book_id)

        language_embed = self.language_embedding(language)
        maturity_rating_embed = self.maturity_rating_embedding(maturity_rating)

        # Embeddings for other categorical features
        author_embed = self.author_proj(authors)
        # category_embed = self.category_embedding(category_label)
        publisher_embed = self.publisher_proj(publisher)
        categories_embed = self.category_proj(categories)

        # Embeddings for other numerical features
        page_count_embed = page_count.unsqueeze(1)
        published_year_embed = published_year.unsqueeze(1)

        # Concatenate all features
        x = torch.cat(
            [
                user_embed,
                book_embed,
                author_embed,
                categories_embed,
                publisher_embed,
                page_count_embed,
                published_year_embed,
                language_embed,
                maturity_rating_embed,
                full_text_embeddings,
            ],
            dim=1,
        )

        # Pass through the fully connected layers
        output = self.fc(x)
        return output.squeeze()
