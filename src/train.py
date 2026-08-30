import json
import os
from pathlib import Path

import torch
import yaml
from torch import nn
from torch.optim import Adam

try:
    from src.dataset import get_dataloaders
    from src.model import get_model
except ModuleNotFoundError:
    from dataset import get_dataloaders
    from model import get_model


def load_config() -> dict:
    config_path = Path(
        os.getenv("TRAINING_CONFIG", "/app/configs/training_config.yaml")
    )
    if not config_path.exists():
        config_path = Path("configs/training_config.yaml")

    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_epoch(model, loader, optimizer, criterion, device, training=True):
    model.train(training)
    total_loss = 0.0
    correct = 0
    total = 0

    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)

        if training:
            optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, targets)

        if training:
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * inputs.size(0)
        correct += outputs.argmax(dim=1).eq(targets).sum().item()
        total += targets.size(0)

    return total_loss / total, correct / total


def main():
    config = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = get_model(
        architecture=config["model"]["architecture"],
        num_classes=config["model"]["num_classes"],
    ).to(device)

    train_loader, val_loader = get_dataloaders(
        data_dir=config["data"]["data_dir"],
        batch_size=config["training"]["batch_size"],
        num_workers=config["training"].get("num_workers", 2),
    )

    optimizer = Adam(
        model.parameters(),
        lr=config["training"]["learning_rate"],
    )
    criterion = nn.CrossEntropyLoss()

    checkpoint_dir = Path(config["output"]["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / config["output"]["model_name"]

    best_val_loss = float("inf")
    patience = config["training"]["early_stopping_patience"]
    patience_counter = 0

    for epoch in range(1, config["training"]["epochs"] + 1):
        train_loss, train_acc = run_epoch(
            model, train_loader, optimizer, criterion, device, training=True
        )
        val_loss, val_acc = run_epoch(
            model, val_loader, optimizer, criterion, device, training=False
        )

        print(json.dumps({
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "train_accuracy": round(train_acc, 4),
            "val_loss": round(val_loss, 4),
            "val_accuracy": round(val_acc, 4),
        }), flush=True)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0

            torch.save({
                "model_state_dict": model.state_dict(),
                "architecture": config["model"]["architecture"],
                "num_classes": config["model"]["num_classes"],
                "class_names": [
                    "airplane", "automobile", "bird", "cat", "deer",
                    "dog", "frog", "horse", "ship", "truck",
                ],
            }, checkpoint_path)

            print(json.dumps({
                "event": "checkpoint_saved",
                "path": str(checkpoint_path),
            }), flush=True)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(json.dumps({
                    "event": "early_stopping",
                    "epoch": epoch,
                }), flush=True)
                break

    print(json.dumps({
        "event": "training_complete",
        "best_val_loss": round(best_val_loss, 4),
    }), flush=True)


if __name__ == "__main__":
    main()
