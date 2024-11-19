import torch
from torch.utils.data import DataLoader
from model import CollaborativeFilteringModel
from trainer import Trainer
import pandas as pd
import pickle


def load_data():
    df = pd.read_csv("../../data/train.csv")
    user_ids = df["user_id"].unique()
    book_ids = df["book_id"].unique()

    user_to_index = {user_id: idx for idx, user_id in enumerate(user_ids)}
    book_to_index = {book_id: idx for idx, book_id in enumerate(book_ids)}

    df["user_id"] = df["user_id"].map(user_to_index)
    df["book_id"] = df["book_id"].map(book_to_index)

    return df, len(user_ids), len(book_ids)


if __name__ == "__main__":
    train_df, num_users, num_books = load_data()

    user_to_index = {
        user_id: idx for idx, user_id in enumerate(train_df["user_id"].unique())
    }
    book_to_index = {
        book_id: idx for idx, book_id in enumerate(train_df["book_id"].unique())
    }

    # Map the IDs to indices in the training data
    train_df["user_id"] = train_df["user_id"].map(user_to_index)
    train_df["book_id"] = train_df["book_id"].map(book_to_index)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = torch.utils.data.TensorDataset(
        torch.tensor(train_df["user_id"].values, dtype=torch.long).to(device),
        torch.tensor(train_df["book_id"].values, dtype=torch.long).to(device),
        torch.tensor(train_df["rating"].values, dtype=torch.float).to(device),
    )
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = CollaborativeFilteringModel(num_users, num_books, embedding_dim=50).to(
        device
    )
    trainer = Trainer(model, dataloader)
    trainer.train(epochs=7)
    trainer.save_model("collaborative_model.pth")

    # save the user_to_index and book_to_index mappings
    with open("user_to_index.pkl", "wb") as f:
        pickle.dump(user_to_index, f)
    with open("book_to_index.pkl", "wb") as f:
        pickle.dump(book_to_index, f)
