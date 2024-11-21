import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler


def cross_validate_model(
    model_class,
    dataset,
    num_users,
    num_books,
    num_authors,
    cat_dim,
    num_publishers,
    num_langs,
    num_folds=5,
    num_epochs=15,
    batch_size=32,
    learning_rate=0.0001,
    device="cuda" if torch.cuda.is_available() else "cpu",
):
    kfold = KFold(n_splits=num_folds, shuffle=True, random_state=42)
    fold_results = []

    # Split dataset into k-folds
    for fold, (train_idx, val_idx) in enumerate(kfold.split(dataset)):
        print(f"Fold {fold+1}/{num_folds}")

        # Create subsets for the current fold
        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)

        # Initialize a new model for each fold with the required parameters
        model = model_class(
            num_users=num_users,
            num_books=num_books,
            num_authors=num_authors,
            cat_dim=cat_dim,
            num_langs=num_langs,
            num_publishers=num_publishers,
        ).to(device)

        # Set up data loaders
        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)

        # Loss function and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        # Training loop
        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0

            for batch in train_loader:
                user_id = batch["user_id"].to(device)
                book_id = batch["book_id"].to(device)
                authors = batch["authors"].to(device)
                categories = batch["categories"].to(device)
                publisher = batch["publisher"].to(device)
                page_count = batch["pageCount"].to(device)
                full_text_embeddings = batch["full_text_embeddings"].to(device)
                published_year = batch["publishedYear"].to(device)
                rating = batch["rating"].to(device)
                language = batch["language"].to(device)
                maturity_rating = batch["maturityRating"].to(device)
                optimizer.zero_grad()

                outputs = model(
                    user_id=user_id,
                    book_id=book_id,
                    authors=authors,
                    categories=categories,
                    publisher=publisher,
                    page_count=page_count,
                    published_year=published_year,
                    full_text_embeddings=full_text_embeddings,
                    language=language,
                    maturity_rating=maturity_rating,
                )

                loss = criterion(outputs, rating)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            print(f"Epoch {epoch+1} - Training Loss: {running_loss/len(train_loader)}")
            # RMSE torch.sqrt(torch.tensor(total_loss / len(train_loader))).item()
            print(
                f"Epoch {epoch+1} - Training RMSE: {torch.sqrt(torch.tensor(running_loss / len(train_loader))).item()}"
            )
            # Validation loop
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    user_id = batch["user_id"].to(device)
                    book_id = batch["book_id"].to(device)
                    authors = batch["authors"].to(device)
                    categories = batch["categories"].to(device)
                    publisher = batch["publisher"].to(device)
                    page_count = batch["pageCount"].to(device)
                    full_text_embeddings = batch["full_text_embeddings"].to(device)
                    published_year = batch["publishedYear"].to(device)
                    rating = batch["rating"].to(device)
                    language = batch["language"].to(device)
                    maturity_rating = batch["maturityRating"].to(device)

                    outputs = model(
                        user_id=user_id,
                        book_id=book_id,
                        authors=authors,
                        categories=categories,
                        publisher=publisher,
                        page_count=page_count,
                        published_year=published_year,
                        full_text_embeddings=full_text_embeddings,
                        language=language,
                        maturity_rating=maturity_rating,
                    )

                    loss = criterion(outputs, rating)
                    val_loss += loss.item()

            print(f"Validation Loss: {val_loss/len(val_loader)}")
            print(
                f"Validation RMSE: {torch.sqrt(torch.tensor(val_loss / len(val_loader))).item()}"
            )

        fold_results.append(val_loss / len(val_loader))

    avg_loss = sum(fold_results) / len(fold_results)
    print(f"Cross-Validation Complete! Average Validation Loss: {avg_loss}")

    return avg_loss
