import torch
import pandas as pd
import ast
from trainer import Trainer
from model import ContentBasedModel
from sklearn.preprocessing import LabelEncoder
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

stemmer = PorterStemmer()


def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    tokens = [stemmer.stem(token) for token in tokens if token.isalnum()]
    return tokens


def load_test_data():
    books_df = pd.read_csv("../../data/extended_books_google_embeddings.csv")
    test_df = pd.read_csv("../../data/test.csv")

    books_df["full_text_embeddings"] = books_df["full_text_embeddings"].apply(
        ast.literal_eval
    )
    books_df["authors"] = books_df["authors"].apply(
        lambda x: x.split(",")[0] if pd.notna(x) else "Unknown"
    )
    author_encoder = LabelEncoder()
    books_df["author_id"] = author_encoder.fit_transform(books_df["authors"])

    books_df["categories"] = books_df["categories"].fillna("").apply(preprocess_text)
    all_categories = {cat for sublist in books_df["categories"] for cat in sublist}
    category_to_id = {
        cat: idx + 1 for idx, cat in enumerate(all_categories)
    }  # +1 for padding
    max_categories_length = 10

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

    merged_df = test_df.merge(books_df, on="book_id")
    book_features = torch.tensor(
        merged_df["full_text_embeddings"].tolist(), dtype=torch.float
    )
    author_features = torch.tensor(merged_df["author_id"].values, dtype=torch.long)
    category_features = torch.tensor(
        merged_df["category_ids"].tolist(), dtype=torch.long
    )
    page_count_features = torch.tensor(merged_df["pageCount"].values, dtype=torch.float)

    return (
        merged_df["id"],
        book_features,
        author_features,
        category_features,
        page_count_features,
    )


model = ContentBasedModel(
    input_dim=768, author_dim=4031, category_dim=1054, embedding_dim=100
)
trainer = Trainer(model, None)
trainer.load(model, "content_based_model.pth")

ids, book_features, author_features, category_features, page_count_features = (
    load_test_data()
)

predictions = trainer.predict(
    book_features, author_features, category_features, page_count_features
)

submission_df = pd.DataFrame({"id": ids, "rating": predictions})
submission_df.to_csv("content_based_submission.csv", index=False)
