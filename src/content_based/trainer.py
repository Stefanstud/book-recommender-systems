import torch
import torch.nn as nn
import torch.optim as optim


class Trainer:
    def __init__(self, model, dataloader, lr=0.001):
        self.model = model
        self.dataloader = dataloader
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(model.parameters(), lr=lr)

    def train(self, epochs=10):
        self.model.train()
        for epoch in range(epochs):
            epoch_loss = 0
            for (
                book_features,
                author_features,
                category_features,
                page_count,
                rating,
            ) in self.dataloader:

                self.optimizer.zero_grad()
                predictions = self.model(
                    book_features, author_features, category_features, page_count
                )
                loss = self.criterion(predictions, rating)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()

            print(
                f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss / len(self.dataloader):.4f}"
            )

    def predict(self, book_features):
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(book_features)
            return predictions.squeeze().tolist()  

    def predict(self, book_features, author_features, category_features, page_count):
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(
                book_features, author_features, category_features, page_count
            )
            return predictions.squeeze().tolist()  

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self, model, path):
        model.load_state_dict(torch.load(path, weights_only=True))
        return model
