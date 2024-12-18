# import torch
# from torch_geometric.data import Data
# import pandas as pd
# from sklearn.preprocessing import LabelEncoder
# import ast


# def load_data():
#     books_file_path = "../../data/extended_books_google_embeddings.csv"
#     books_data = pd.read_csv(books_file_path)
#     train_file_path = "../../data/train.csv"
#     train_data = pd.read_csv(train_file_path)

#     df = train_data.merge(books_data, on="book_id", how="left")

#     unique_user_ids = df["user_id"].unique()
#     user_id_map = {
#         original_id: new_id for new_id, original_id in enumerate(unique_user_ids)
#     }
#     df["user_id_mapped"] = df["user_id"].map(user_id_map)

#     unique_book_ids = df["book_id"].unique()
#     book_id_map = {
#         original_id: new_id for new_id, original_id in enumerate(unique_book_ids)
#     }
#     df["book_id_mapped"] = df["book_id"].map(book_id_map)

#     num_users = len(unique_user_ids)
#     num_books = len(unique_book_ids)
#     num_nodes = num_users + num_books

#     print("Number of unique users:", num_users)
#     print("Number of unique books:", num_books)

#     # Adjust book IDs to be distinct from user IDs
#     df["book_id_mapped"] += num_users

#     # Create edge indices
#     edge_index = (
#         torch.tensor(df[["user_id_mapped", "book_id_mapped"]].values).t().contiguous()
#     )

#     ratings = torch.tensor(df["rating"].values, dtype=torch.float32)

#     books_data = books_data[books_data["book_id"].isin(unique_book_ids)].copy()
#     books_data["book_id_mapped"] = books_data["book_id"].map(book_id_map)
#     books_data.sort_values("book_id_mapped", inplace=True)

#     # Features from books_data
#     page_count = torch.tensor(
#         books_data["pageCount"].fillna(0).values, dtype=torch.float32
#     )
#     ratings_count = torch.tensor(
#         books_data["ratingsCount"].fillna(0).values, dtype=torch.float32
#     )

#     maturity_rating_encoded = LabelEncoder().fit_transform(
#         books_data["maturityRating"].fillna("Unknown")
#     )
#     language_encoded = LabelEncoder().fit_transform(
#         books_data["language"].fillna("Unknown")
#     )
#     publisher_encoded = LabelEncoder().fit_transform(
#         books_data["publisher"].fillna("Unknown")
#     )

#     maturity_rating = torch.tensor(maturity_rating_encoded, dtype=torch.float32)
#     language = torch.tensor(language_encoded, dtype=torch.float32)
#     publisher = torch.tensor(publisher_encoded, dtype=torch.float32)

#     published_date = pd.to_datetime(books_data["publishedDate"], errors="coerce")
#     published_year = torch.tensor(
#         published_date.dt.year.fillna(0).values, dtype=torch.float32
#     )

#     def parse_embedding(embedding_str):
#         return torch.tensor(ast.literal_eval(embedding_str), dtype=torch.float32)

#     # full_text_embeddings_list = (
#     #     books_data["full_text_embeddings"].apply(parse_embedding).tolist()
#     # )
#     # full_text_embeddings = torch.stack(full_text_embeddings_list)

#     # first_authors = (
#     #     books_data["authors"]
#     #     .fillna("Unknown")
#     #     .apply(lambda x: x.split(",")[0].strip() if x else "Unknown")
#     # )

#     # author_encoder = LabelEncoder()
#     # first_author_encoded = author_encoder.fit_transform(first_authors)
#     # first_author = torch.tensor(first_author_encoded, dtype=torch.float32)

#     # num_authors = len(author_encoder.classes_)
#     # print("Number of unique first authors:", num_authors)

#     book_features = torch.stack(
#         [
#             page_count,
#             ratings_count,
#             maturity_rating,
#             published_year,
#             language,
#             publisher,
#             # first_author,
#             # Add other features like full_text_embeddings if needed
#         ],
#         dim=1,
#     )

#     feature_dim = book_features.size(1)

#     x = torch.zeros((num_nodes, feature_dim), dtype=torch.float32)

#     book_indices = books_data["book_id_mapped"].values
#     x[book_indices] = book_features

#     data = Data(edge_index=edge_index, x=x, y=ratings, num_nodes=num_nodes)

#     return data, num_users, num_books, feature_dim


import torch
from torch_geometric.data import Data
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import ast


def load_data():
    books_file_path = "../../data/extended_books_google_embeddings.csv"
    books_data = pd.read_csv(books_file_path)
    train_file_path = "../../data/train.csv"
    train_data = pd.read_csv(train_file_path)

    df = train_data.merge(books_data, on="book_id", how="left")
    unique_user_ids = df["user_id"].unique()
    user_id_map = {
        original_id: new_id for new_id, original_id in enumerate(unique_user_ids)
    }
    df["user_id_mapped"] = df["user_id"].map(user_id_map)

    unique_book_ids = df["book_id"].unique()
    book_id_map = {
        original_id: new_id for new_id, original_id in enumerate(unique_book_ids)
    }
    df["book_id_mapped"] = df["book_id"].map(book_id_map)
    num_users = len(unique_user_ids)
    num_books = len(unique_book_ids)
    num_nodes = num_users + num_books

    print("Number of unique users:", num_users)
    print("Number of unique books:", num_books)

    df["book_id_mapped"] += num_users
    edge_index = (
        torch.tensor(df[["user_id_mapped", "book_id_mapped"]].values).t().contiguous()
    )

    ratings = torch.tensor(df["rating"].values, dtype=torch.float32)

    books_data = books_data[books_data["book_id"].isin(unique_book_ids)].copy()
    books_data["book_id_mapped"] = books_data["book_id"].map(book_id_map)
    books_data.sort_values("book_id_mapped", inplace=True)

    page_count = torch.tensor(
        books_data["pageCount"].fillna(0).values, dtype=torch.float32
    )
    ratings_count = torch.tensor(
        books_data["ratingsCount"].fillna(0).values, dtype=torch.float32
    )
    maturity_rating_encoded = LabelEncoder().fit_transform(
        books_data["maturityRating"].fillna("Unknown")
    )
    language_encoded = LabelEncoder().fit_transform(
        books_data["language"].fillna("Unknown")
    )
    publisher_encoded = LabelEncoder().fit_transform(
        books_data["publisher"].fillna("Unknown")
    )
    maturity_rating = torch.tensor(maturity_rating_encoded, dtype=torch.float32)
    language = torch.tensor(language_encoded, dtype=torch.float32)
    publisher = torch.tensor(publisher_encoded, dtype=torch.float32)
    published_date = pd.to_datetime(books_data["publishedDate"], errors="coerce")
    published_year = torch.tensor(
        published_date.dt.year.fillna(0).values, dtype=torch.float32
    )

    book_features = torch.stack(
        [
            publisher,
        ],
        dim=1,
    )

    feature_dim = book_features.size(1)
    x = torch.zeros((num_nodes, feature_dim), dtype=torch.float32)
    book_indices = books_data["book_id_mapped"].values
    x[book_indices] = book_features

    data = Data(edge_index=edge_index, x=x, y=ratings, num_nodes=num_nodes)
    return data, num_users, num_books, feature_dim, user_id_map, book_id_map
