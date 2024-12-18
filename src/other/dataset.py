import pandas as pd
import torch
from torch.utils.data import Dataset
import string
import numpy as np

# from sklearn.impute import KNNImputer
from collections import Counter
import ast
from nltk.stem import PorterStemmer
import nltk
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MultiLabelBinarizer
from sentence_transformers import SentenceTransformer

nltk.download("punkt")


class BooksDataset(Dataset):
    def __init__(self, dataset):
        self.books = dataset.copy()
        # self.books = self.books.drop(columns=["textSnippet"])
        # self.imputer = KNNImputer(n_neighbors=5)
        self.stemmer = PorterStemmer()
        self.category_model = SentenceTransformer("thenlper/gte-small")
        self._preprocess()

    def _preprocess(self):
        # Handle missing numerical values
        # fill with median
        self.books["pageCount"] = self.books["pageCount"].fillna(
            self.books["pageCount"].median()
        )
        print("Numerical columns imputed")

        self.books["full_text_embeddings"] = self.books["full_text_embeddings"].apply(
            eval
        )  # TODO: change how we save it so its not a string
        self.books = self.books.drop(
            columns=["title", "subtitle", "description", "full_text"]
        )
        print("Embeddings loaded")

        # Handle dates
        self.books["publishedDate"] = pd.to_datetime(
            self.books["publishedDate"], errors="coerce"
        )
        self.books["publishedYear"] = self.books["publishedDate"].dt.year
        self.books["publishedYear"] = self.books["publishedYear"].fillna(
            self.books["publishedYear"].median()
        )
        self.books = self.books.drop(columns=["publishedDate"])

        print("Dates handled")

        # Fill missing categorical values
        # fill missing publisher with "Unknown" and convert to numerical
        self.books["publisher"] = self.books["publisher"].fillna("Unknown")
        self.books["publisher"] = self.books["publisher"].str.split(",")

        publisher_enc = MultiLabelBinarizer()
        pub_encoded = publisher_enc.fit_transform(self.books["publisher"].values)
        self.books["publisher"] = pub_encoded.tolist()

        # fill missing maturityRating with most frequent value and convert to numerical
        self.books["maturityRating"] = self.books["maturityRating"].fillna(
            self.books["maturityRating"].mode().values[0]
        )

        # 0 for "NOT_MATURE" and 1 for "MATURE"
        self.books["maturityRating"] = self.books["maturityRating"].apply(
            lambda x: 0 if x == "NOT_MATURE" else 1
        )

        # fill missing language with most frequent value and convert to numerical
        self.books["language"] = self.books["language"].fillna(
            self.books["language"].mode().values[0]
        )
        lang_enc = LabelEncoder()
        self.books["language"] = lang_enc.fit_transform(self.books["language"])

        # Handle authors: Keep only the first author
        # input missing authors with "Unknown"
        self.books["authors"] = self.books["authors"].fillna("Unknown")
        self.books["authors"] = self.books["authors"].str.split(",")

        authors_enc = MultiLabelBinarizer()
        authors_encoded = authors_enc.fit_transform(self.books["authors"].values)
        self.books["authors"] = authors_encoded.tolist()

        # Handle categories: Remove punctuation, split, and stem
        self.books["categories"] = self.books["categories"].fillna("Unknown")
        # split by ,
        self.books["categories"] = self.books["categories"].apply(
            lambda x: x.split(",")
        )
        self.category_vocab = self._build_category_vocab(self.books["categories"])
        # get the mapping of each category do the emb dim and average if there are multiple categories
        self.books["categories"] = self.books["categories"].apply(
            lambda x: torch.tensor([self.category_vocab[cat] for cat in x]).mean(dim=0)
        )

        # standardize all features
        self.books["pageCount"] = (
            self.books["pageCount"] - self.books["pageCount"].mean()
        ) / self.books["pageCount"].std()
        self.books["publishedYear"] = (
            self.books["publishedYear"] - self.books["publishedYear"].mean()
        ) / self.books["publishedYear"].std()

    def _build_category_vocab(self, categories_list):
        all_categories = [
            category for categories in categories_list for category in categories
        ]
        set_categories = set(all_categories)
        vocab = {}
        for index, category in enumerate(set_categories):
            vocab[category] = self.category_model.encode(category)

        return vocab

    def __len__(self):
        return len(self.books)

    def __getitem__(self, idx):
        row = self.books.iloc[idx]
        features = {
            "user_id": torch.tensor(row["user_id"], dtype=torch.long),
            "book_id": torch.tensor(row["book_id"], dtype=torch.long),
            "pageCount": torch.tensor(row["pageCount"], dtype=torch.float32),
            "authors": torch.tensor(
                row["authors"], dtype=torch.float32
            ),  # k project to smaller dim
            "categories": torch.tensor(
                row["categories"], dtype=torch.float32
            ),  # emb - project to smaller dim
            "publishedYear": torch.tensor(row["publishedYear"], dtype=torch.float32),
            "publisher": torch.tensor(
                row["publisher"], dtype=torch.float32
            ),  # k - multiple publishers per book - project to smaller dim
            "maturityRating": torch.tensor(
                row["maturityRating"], dtype=torch.long
            ),  # b - embed with not big dimensionality
            "language": torch.tensor(row["language"], dtype=torch.long),  # c embed
            "full_text_embeddings": torch.tensor(
                row["full_text_embeddings"], dtype=torch.float32
            ),  # dont do anything
            "rating": torch.tensor(row["rating"], dtype=torch.float32),
        }
        return features
