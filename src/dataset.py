import pandas as pd
import torch
from torch.utils.data import Dataset
import string

# from sklearn.impute import KNNImputer
from collections import Counter
import ast
from nltk.stem import PorterStemmer
import nltk
from sklearn.preprocessing import LabelEncoder  # , MultiLabelBinarizer

nltk.download("punkt")


class BooksDataset(Dataset):
    def __init__(self, dataset):
        self.books = dataset.copy()
        # self.books = self.books.drop(columns=["textSnippet"])

        # self.imputer = KNNImputer(n_neighbors=5)
        self.stemmer = PorterStemmer()
        self._preprocess()

    def _preprocess(self):
        # create masks for missing data
        self.books["pageCount_missing"] = self.books["pageCount"].isnull().astype(int)
        self.books["description_missing"] = (
            self.books["description"].isnull().astype(int)
        )
        self.books["categories_missing"] = self.books["categories"].isnull().astype(int)
        self.books["averageRating_missing"] = (
            self.books["averageRating"].isnull().astype(int)
        )
        self.books["ratingsCount_missing"] = (
            self.books["ratingsCount"].isnull().astype(int)
        )
        self.books["publisher_missing"] = self.books["publisher"].isnull().astype(int)

        # Handle missing numerical values
        # fill with median
        self.books["pageCount"] = self.books["pageCount"].fillna(
            self.books["pageCount"].median()
        )
        self.books["averageRating"] = self.books["averageRating"].fillna(
            self.books["averageRating"].median()
        )
        self.books["ratingsCount"] = self.books["ratingsCount"].fillna(
            self.books["ratingsCount"].median()
        )

        print("Numerical columns imputed")

        self.books["full_text_embeddings"] = self.books["full_text_embeddings"].apply(
            eval
        )
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
        pub_enc = LabelEncoder()
        self.books["publisher"] = pub_enc.fit_transform(self.books["publisher"])

        # fill missing maturityRating with most frequent value and convert to numerical
        self.books["maturityRating"] = self.books["maturityRating"].fillna(
            self.books["maturityRating"].mode().values[0]
        )
        self.books["maturityRating"] = (
            self.books["maturityRating"].astype("category").cat.codes
        )

        # fill missing language with most frequent value and convert to numerical
        self.books["language"] = self.books["language"].fillna(
            self.books["language"].mode().values[0]
        )
        lang_enc = LabelEncoder()
        self.books["language"] = lang_enc.fit_transform(self.books["language"])

        # Handle authors: Keep only the first author
        self.books["authors"] = (
            self.books["authors"].fillna("[]").apply(self._get_first_author)
        )
        self.author_vocab = self._build_author_vocab(self.books["authors"])
        self.books["author_label"] = self.books["authors"].apply(
            lambda authors: self.author_vocab.get(
                authors[0], self.author_vocab["<unk>"]
            )
        )

        # Handle categories: Remove punctuation, split, and stem
        self.books["categories"] = (
            self.books["categories"].fillna("[]").apply(self._preprocess_categories)
        )
        self.category_vocab = self._build_category_vocab(self.books["categories"])
        self.max_categories_len = self.books["categories"].apply(len).max()
        self.books["category_label"] = self.books["categories"].apply(
            lambda categories: self._encode_and_pad_categories(
                categories, self.max_categories_len
            )
        )

        # standardize all features
        self.books["pageCount"] = (
            self.books["pageCount"] - self.books["pageCount"].mean()
        ) / self.books["pageCount"].std()
        self.books["averageRating"] = (
            self.books["averageRating"] - self.books["averageRating"].mean()
        ) / self.books["averageRating"].std()
        self.books["ratingsCount"] = (
            self.books["ratingsCount"] - self.books["ratingsCount"].mean()
        ) / self.books["ratingsCount"].std()
        self.books["publishedYear"] = (
            self.books["publishedYear"] - self.books["publishedYear"].mean()
        ) / self.books["publishedYear"].std()

    def _get_first_author(self, x):
        if isinstance(x, str):
            x = [x]
        elif isinstance(x, list) and len(x) > 0:
            x = [x[0]]
        else:
            x = ["<pad>"]
        return x

    def _build_author_vocab(self, authors_list):
        all_authors = [
            author for authors in authors_list for author in authors if author
        ]
        author_counter = Counter(all_authors)
        author_vocab = {"<pad>": 0, "<unk>": 1}
        for index, author in enumerate(author_counter.keys(), start=2):
            author_vocab[author] = index
        return author_vocab

    def _preprocess_categories(self, category_list):
        if isinstance(category_list, str):
            try:
                category_list = ast.literal_eval(category_list)
            except (ValueError, SyntaxError):
                category_list = [category_list]
        processed_categories = []
        for category in category_list:
            category = category.translate(str.maketrans("", "", string.punctuation))
            words = category.split()
            stemmed_words = [self.stemmer.stem(word.lower()) for word in words]
            processed_categories.extend(stemmed_words)
        return processed_categories

    def _build_category_vocab(self, categories_list):
        all_categories = [word for categories in categories_list for word in categories]
        category_counter = Counter(all_categories)
        category_vocab = {"<pad>": 0, "<unk>": 1}
        for index, word in enumerate(category_counter.keys(), start=2):
            category_vocab[word] = index
        return category_vocab

    def _encode_and_pad_categories(self, categories, max_len):
        pad_index = self.category_vocab["<pad>"]
        encoded_categories = [
            self.category_vocab.get(word, self.category_vocab["<unk>"])
            for word in categories
        ]
        if len(encoded_categories) < max_len:
            encoded_categories += [pad_index] * (max_len - len(encoded_categories))
        else:
            encoded_categories = encoded_categories[:max_len]
        return encoded_categories

    def __len__(self):
        return len(self.books)

    def __getitem__(self, idx):
        row = self.books.iloc[idx]
        features = {
            "user_id": torch.tensor(row["user_id"], dtype=torch.long),
            "book_id": torch.tensor(row["book_id"], dtype=torch.long),
            "pageCount": torch.tensor(row["pageCount"], dtype=torch.float32),
            "averageRating": torch.tensor(row["averageRating"], dtype=torch.float32),
            "ratingsCount": torch.tensor(row["ratingsCount"], dtype=torch.float32),
            "author_label": torch.tensor(row["author_label"], dtype=torch.long),
            "category_label": torch.tensor(row["category_label"], dtype=torch.long),
            "publishedYear": torch.tensor(row["publishedYear"], dtype=torch.float32),
            "publisher": torch.tensor(row["publisher"], dtype=torch.long),
            "maturityRating": torch.tensor(row["maturityRating"], dtype=torch.long),
            "language": torch.tensor(row["language"], dtype=torch.long),
            "publisher_missing": torch.tensor(
                row["publisher_missing"], dtype=torch.bool
            ),
            "pageCount_missing": torch.tensor(
                row["pageCount_missing"], dtype=torch.bool
            ),
            "description_missing": torch.tensor(
                row["description_missing"], dtype=torch.bool
            ),
            "categories_missing": torch.tensor(
                row["categories_missing"], dtype=torch.bool
            ),
            "averageRating_missing": torch.tensor(
                row["averageRating_missing"], dtype=torch.bool
            ),
            "ratingsCount_missing": torch.tensor(
                row["ratingsCount_missing"], dtype=torch.bool
            ),
            "full_text_embeddings": torch.tensor(
                row["full_text_embeddings"], dtype=torch.float32
            ),
            "rating": torch.tensor(row["rating"], dtype=torch.float32),
        }
        return features
