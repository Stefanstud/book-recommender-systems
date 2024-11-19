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
            for user_id, book_id, rating in self.dataloader:
                self.optimizer.zero_grad()
                predictions = self.model(user_id, book_id)
                loss = self.criterion(predictions, rating)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()

            # rmse error
            rmse = torch.sqrt(torch.tensor(epoch_loss / len(self.dataloader)))

            print(
                f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss / len(self.dataloader) :.4f} RMSE Error: {rmse:.4f}"
            )

    def predict(self, user_id, book_id):
        self.model.eval()
        with torch.no_grad():
            user_id_tensor = torch.tensor(user_id, dtype=torch.long)
            book_id_tensor = torch.tensor(book_id, dtype=torch.long)

            if user_id_tensor.dim() == 0:
                user_id_tensor = user_id_tensor.unsqueeze(0)
            if book_id_tensor.dim() == 0:
                book_id_tensor = book_id_tensor.unsqueeze(0)

            predictions = self.model(user_id_tensor, book_id_tensor)
            return predictions.squeeze().item()

    def save_model(self, file_path="collaborative_model.pth"):
        torch.save(self.model.state_dict(), file_path)
        print(f"Model saved to {file_path}")

    def load_model(self, model, file_path="collaborative_model.pth"):
        model.load_state_dict(torch.load(file_path, weights_only=True))
        model.eval()
        print(f"Model loaded from {file_path}")
