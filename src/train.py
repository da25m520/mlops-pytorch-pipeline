import json
import os
import sys

import torch
import torch.nn as nn
import yaml

# Make the project root importable when running:
# python src/train.py
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.dataset import get_dataloaders
from src.model import get_model


def load_config(config_path="configs/training_config.yaml"):
    with open(config_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def evaluate(model, dataloader, criterion, device):
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)

            predictions = torch.argmax(outputs, dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    average_loss = total_loss / total
    accuracy = correct / total

    return average_loss, accuracy


def train_model(config_path="configs/training_config.yaml"):
    config = load_config(config_path)

    # -----------------------------
    # Configuration
    # -----------------------------
    architecture = config["model"]["architecture"]
    num_classes = config["model"]["num_classes"]

    epochs = config["training"]["epochs"]
    batch_size = config["training"]["batch_size"]
    learning_rate = config["training"]["learning_rate"]
    patience = config["training"]["early_stopping_patience"]

    data_dir = config["data"]["data_dir"]

    checkpoint_dir = config["output"]["checkpoint_dir"]
    model_name = config["output"]["model_name"]

    # Convert relative paths to project-relative paths.
    if not os.path.isabs(data_dir):
        data_dir = os.path.join(PROJECT_ROOT, data_dir)

    if not os.path.isabs(checkpoint_dir):
        checkpoint_dir = os.path.join(PROJECT_ROOT, checkpoint_dir)

    os.makedirs(checkpoint_dir, exist_ok=True)

    checkpoint_path = os.path.join(
        checkpoint_dir,
        model_name,
    )

    # -----------------------------
    # Device
    # -----------------------------
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(
        json.dumps(
            {
                "event": "training_started",
                "device": str(device),
                "architecture": architecture,
                "epochs": epochs,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
            }
        )
    )

    # -----------------------------
    # Data
    # -----------------------------
    train_loader, val_loader = get_dataloaders(
        data_dir=data_dir,
        batch_size=batch_size,
        num_workers=0,
    )

    # -----------------------------
    # Model
    # -----------------------------
    model = get_model(
        architecture=architecture,
        num_classes=num_classes,
    ).to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
    )

    # -----------------------------
    # Early stopping
    # -----------------------------
    best_val_loss = float("inf")
    epochs_without_improvement = 0

    # -----------------------------
    # Training loop
    # -----------------------------
    for epoch in range(epochs):
        model.train()

        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

            predictions = torch.argmax(outputs, dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

        train_loss = running_loss / total
        train_accuracy = correct / total

        val_loss, val_accuracy = evaluate(
            model,
            val_loader,
            criterion,
            device,
        )

        # Structured JSON-lines logging.
        print(
            json.dumps(
                {
                    "epoch": epoch + 1,
                    "train_loss": round(train_loss, 6),
                    "train_accuracy": round(train_accuracy, 6),
                    "val_loss": round(val_loss, 6),
                    "val_accuracy": round(val_accuracy, 6),
                }
            )
        )

        # -----------------------------
        # Save best checkpoint
        # -----------------------------
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "architecture": architecture,
                    "num_classes": num_classes,
                    "epoch": epoch + 1,
                    "val_loss": val_loss,
                    "val_accuracy": val_accuracy,
                },
                checkpoint_path,
            )

            print(
                json.dumps(
                    {
                        "event": "checkpoint_saved",
                        "path": checkpoint_path,
                        "epoch": epoch + 1,
                        "val_loss": round(val_loss, 6),
                        "val_accuracy": round(val_accuracy, 6),
                    }
                )
            )

        else:
            epochs_without_improvement += 1

        # -----------------------------
        # Early stopping
        # -----------------------------
        if epochs_without_improvement >= patience:
            print(
                json.dumps(
                    {
                        "event": "early_stopping",
                        "epoch": epoch + 1,
                        "patience": patience,
                    }
                )
            )
            break

    print(
        json.dumps(
            {
                "event": "training_completed",
                "checkpoint": checkpoint_path,
            }
        )
    )

    return checkpoint_path


def main():
    config_path = "configs/training_config.yaml"

    if len(sys.argv) > 1:
        config_path = sys.argv[1]

    train_model(config_path)


if __name__ == "__main__":
    main()