import torch
import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, TensorDataset
import ast
from model import ContentBasedModel
from trainer import Trainer
import sys


stemmer = PorterStemmer()


def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    tokens = [stemmer.stem(token) for token in tokens if token.isalnum()]
    return tokens


def load_data():
    books_df = pd.read_csv("../../data/extended_books_google_embeddings.csv")
    train_df = pd.read_csv("../../data/train.csv")

    # Process embeddings
    books_df["full_text_embeddings"] = books_df["full_text_embeddings"].apply(
        ast.literal_eval
    )

    # Process authors: Take the first author only
    books_df["authors"] = books_df["authors"].apply(
        lambda x: x.split(",")[0] if pd.notna(x) else "Unknown"
    )
    author_encoder = LabelEncoder()
    books_df["author_id"] = author_encoder.fit_transform(books_df["authors"])

    # Process categories: Preprocess, encode, and pad
    books_df["categories"] = books_df["categories"].fillna("").apply(preprocess_text)
    all_categories = {cat for sublist in books_df["categories"] for cat in sublist}
    category_to_id = {cat: idx + 1 for idx, cat in enumerate(all_categories)}
    max_categories_length = 10  # Fixed length for padding

    def encode_and_pad_categories(categories):
        encoded = [category_to_id.get(cat, 0) for cat in categories]
        if len(encoded) < max_categories_length:
            encoded += [0] * (max_categories_length - len(encoded))
        return encoded[:max_categories_length]

    books_df["category_ids"] = books_df["categories"].apply(encode_and_pad_categories)
    books_df["pageCount"] = books_df["pageCount"].fillna(0)
    page_count_mean = books_df["pageCount"].mean()
    page_count_std = books_df["pageCount"].std()
    books_df["pageCount"] = (books_df["pageCount"] - page_count_mean) / page_count_std

    # Merge train data with book features
    merged_df = train_df.merge(books_df, on="book_id")

    # Extract features after merging
    book_features = torch.tensor(
        merged_df["full_text_embeddings"].tolist(), dtype=torch.float
    )
    author_features = torch.tensor(merged_df["author_id"].values, dtype=torch.long)
    category_features = torch.tensor(
        merged_df["category_ids"].tolist(), dtype=torch.long
    )
    page_count_features = torch.tensor(merged_df["pageCount"].values, dtype=torch.float)
    ratings = torch.tensor(merged_df["rating"].values, dtype=torch.float)

    # Create a DataLoader
    dataset = TensorDataset(
        book_features, author_features, category_features, page_count_features, ratings
    )
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Define input dimensions
    input_dims = {
        "embedding_dim": book_features.shape[1],  # 768 for text embeddings
        "author_dim": len(author_encoder.classes_),  # Number of unique authors
        "category_dim": len(category_to_id) + 1,  # +1 for padding
    }

    return dataloader, input_dims


if __name__ == "__main__":
    dataloader, input_dim = load_data()
    # print(input_dim)
    # sys.exit(0)
    model = ContentBasedModel(
        input_dim=input_dim["embedding_dim"],
        author_dim=input_dim["author_dim"],
        category_dim=input_dim["category_dim"],
        embedding_dim=100,
    )
    trainer = Trainer(model, dataloader)
    trainer.train(epochs=40)
    trainer.save("content_based_model.pth")
