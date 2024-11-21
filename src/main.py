import pandas as pd
from model import RecommenderModel
from wide_deep import WideAndDeepModel
from trainer import cross_validate_model
from dataset import BooksDataset
from sklearn.preprocessing import StandardScaler


def main():
    books_file_path = "../data/extended_books_google_embeddings.csv"
    books_data = pd.read_csv(books_file_path)

    train_file_path = "../data/train.csv"
    train_data = pd.read_csv(train_file_path)

    # merge on book_id column
    merged_data = train_data.merge(books_data, on="book_id", how="left")

    books_dataset = BooksDataset(merged_data)
    print(f"Number of samples in dataset: {len(books_dataset)}")

    # Calculate the number of unique values needed for the model
    num_users = books_dataset.books["user_id"].max() + 1
    num_books = books_dataset.books["book_id"].max() + 1
    num_authors = len(books_dataset.books["authors"][0])
    num_publishers = len(books_dataset.books["publisher"][0])
    num_langs = books_dataset.books["language"].max() + 1
    cat_dim = 384

    # Fwross-validation on the model
    avg_loss = cross_validate_model(
        model_class=RecommenderModel,
        dataset=books_dataset,
        num_authors=num_authors,
        num_users=num_users,
        num_books=num_books,
        cat_dim=cat_dim,
        num_publishers=num_publishers,
        num_langs=num_langs,
    )
    print(f"Average RMSE from Cross-Validation: {avg_loss}")


if __name__ == "__main__":
    main()
