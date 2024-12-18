import torch
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.preprocessing import LabelEncoder
from graph_model import GraphRecommendationModel
from load_data import load_data

data, num_users, num_books, feature_dim, user_id_map, book_id_map = load_data()
test_file_path = "../../data/test.csv"
test_df = pd.read_csv(test_file_path)

test_df["user_id_mapped"] = test_df["user_id"].map(user_id_map)
test_df["book_id_mapped"] = test_df["book_id"].map(book_id_map)
test_df["book_id_mapped"] += num_users  # Adjust book IDs

unmapped_users = test_df["user_id_mapped"].isnull()
unmapped_books = test_df["book_id_mapped"].isnull()

print(f"Unmapped users in test set: {unmapped_users.sum()}")
print(f"Unmapped books in test set: {unmapped_books.sum()}")

if unmapped_users.any():
    new_user_ids = test_df.loc[unmapped_users, "user_id"].unique()
    new_user_id_map = {
        original_id: num_users + idx for idx, original_id in enumerate(new_user_ids)
    }
    user_id_map.update(new_user_id_map)
    test_df.loc[unmapped_users, "user_id_mapped"] = test_df.loc[
        unmapped_users, "user_id"
    ].map(user_id_map)
    num_new_users = len(new_user_ids)
    num_users += num_new_users

if unmapped_books.any():
    new_book_ids = test_df.loc[unmapped_books, "book_id"].unique()
    new_book_id_map = {
        original_id: num_books + idx for idx, original_id in enumerate(new_book_ids)
    }
    book_id_map.update(new_book_id_map)
    test_df.loc[unmapped_books, "book_id_mapped"] = test_df.loc[
        unmapped_books, "book_id"
    ].map(book_id_map)
    num_new_books = len(new_book_ids)
    num_books += num_new_books

num_nodes = num_users + num_books
feature_dim = data.x.size(1)
x = torch.zeros((num_nodes, feature_dim), dtype=torch.float32)
x[: data.x.size(0)] = data.x  

if unmapped_books.any():
    books_file_path = "../../data/extended_books_google_embeddings.csv"
    books_data = pd.read_csv(books_file_path)

    new_books_data = books_data[books_data["book_id"].isin(new_book_ids)].copy()
    new_books_data["book_id_mapped"] = new_books_data["book_id"].map(book_id_map)
    new_books_data.sort_values("book_id_mapped", inplace=True)

    page_count = torch.tensor(
        new_books_data["pageCount"].fillna(0).values, dtype=torch.float32
    )
    ratings_count = torch.tensor(
        new_books_data["ratingsCount"].fillna(0).values, dtype=torch.float32
    )
    maturity_rating_encoded = LabelEncoder().fit_transform(
        new_books_data["maturityRating"].fillna("Unknown")
    )
    language_encoded = LabelEncoder().fit_transform(
        new_books_data["language"].fillna("Unknown")
    )
    publisher_encoded = LabelEncoder().fit_transform(
        new_books_data["publisher"].fillna("Unknown")
    )
    maturity_rating = torch.tensor(maturity_rating_encoded, dtype=torch.float32)
    language = torch.tensor(language_encoded, dtype=torch.float32)
    publisher = torch.tensor(publisher_encoded, dtype=torch.float32)
    published_date = pd.to_datetime(new_books_data["publishedDate"], errors="coerce")
    published_year = torch.tensor(
        published_date.dt.year.fillna(0).values, dtype=torch.float32
    )

    new_book_features = torch.stack(
        [
            page_count,
            ratings_count,
            maturity_rating,
            published_year,
            language,
            publisher,
        ],
        dim=1,
    )

    new_book_indices = new_books_data["book_id_mapped"].values
    x[new_book_indices] = new_book_features

edge_index = torch.tensor(
    test_df[["user_id_mapped", "book_id_mapped"]].values.T, dtype=torch.long
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
embedding_dim = 32  
model = GraphRecommendationModel(num_users, num_books, feature_dim, embedding_dim)
model.load_state_dict(torch.load("graph_model.pth", map_location=device))
model.to(device)
model.eval()

src_nodes = torch.tensor(test_df["user_id_mapped"].values, dtype=torch.long)
dst_nodes = torch.tensor(test_df["book_id_mapped"].values, dtype=torch.long)

x = x.to(device)
edge_index = edge_index.to(device)
src_nodes = src_nodes.to(device)
dst_nodes = dst_nodes.to(device)

test_dataset = TensorDataset(src_nodes, dst_nodes)
test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=False)
all_predictions = []

with torch.no_grad():
    for batch_src_nodes, batch_dst_nodes in test_loader:
        batch_src_nodes = batch_src_nodes.to(device)
        batch_dst_nodes = batch_dst_nodes.to(device)

        predictions = model(edge_index, x, batch_src_nodes, batch_dst_nodes)
        all_predictions.append(predictions.cpu())

all_predictions = torch.cat(all_predictions).numpy()
all_predictions = np.clip(all_predictions, 1, 5)
test_df["rating"] = all_predictions
submission = test_df[["id", "rating"]]
submission.to_csv("submission.csv", index=False)
