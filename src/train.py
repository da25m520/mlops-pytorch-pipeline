import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.dataset import SimpleDataset
from src.model import SimpleClassifier


def train_model(X, y, epochs=10, batch_size=16, learning_rate=0.001):
    dataset = SimpleDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = SimpleClassifier()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    model.train()

    for epoch in range(epochs):
        total_loss = 0.0

        for features, labels in dataloader:
            optimizer.zero_grad()

            outputs = model(features)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        average_loss = total_loss / len(dataloader)

        print(
            f"Epoch [{epoch + 1}/{epochs}], "
            f"Loss: {average_loss:.4f}"
        )

    return model


if __name__ == "__main__":
    import numpy as np

    X = np.array([
        [1.0, 2.0, 3.0, 4.0],
        [2.0, 3.0, 4.0, 5.0],
        [5.0, 6.0, 7.0, 8.0],
        [6.0, 7.0, 8.0, 9.0],
    ])

    y = np.array([0, 0, 1, 1])

    train_model(X, y)