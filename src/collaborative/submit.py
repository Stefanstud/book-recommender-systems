import pandas as pd
import pickle
from trainer import Trainer
from model import CollaborativeFilteringModel

# Load the test and train data
test_df = pd.read_csv("../../data/test.csv")
train_df = pd.read_csv("../../data/train.csv")

# Load the saved mappings from training
with open("user_to_index.pkl", "rb") as f:
    user_to_index = pickle.load(f)
with open("book_to_index.pkl", "rb") as f:
    book_to_index = pickle.load(f)

test_df["user_id"] = (
    test_df["user_id"].map(user_to_index).fillna(len(user_to_index)).astype(int)
)
test_df["book_id"] = (
    test_df["book_id"].map(book_to_index).fillna(len(book_to_index)).astype(int)
)

# Get the number of unique users and books from the training data
num_users = train_df["user_id"].nunique()
num_books = train_df["book_id"].nunique()

model = CollaborativeFilteringModel(num_users, num_books, embedding_dim=50)
trainer = Trainer(model, None)
trainer.load_model(model, "collaborative_model.pth")

predictions = []
for index, row in test_df.iterrows():
    user_id = row["user_id"]
    book_id = row["book_id"]
    predicted_rating = trainer.predict(user_id, book_id)
    predictions.append(predicted_rating)

test_df["rating"] = predictions
submission_df = test_df[["id", "rating"]]
submission_df.to_csv("collaborative_submission.csv", index=False)
