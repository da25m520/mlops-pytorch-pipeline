import os
from io import BytesIO
from pathlib import Path

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image

try:
    from src.model import get_model
except ModuleNotFoundError:
    from model import get_model
from torchvision import transforms


app = FastAPI(title="PyTorch CIFAR-10 Model API")

CHECKPOINT_PATH = Path(
    os.getenv("MODEL_PATH", "/app/checkpoints/classifier_v1.pt")
)

CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]

model = None


def load_model():
    global model
    if not CHECKPOINT_PATH.exists():
        return

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location="cpu",
        weights_only=False,
    )
    model = get_model(
        architecture=checkpoint.get("architecture", "simple_cnn"),
        num_classes=checkpoint.get("num_classes", 10),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()


@app.on_event("startup")
def startup_event():
    load_model()


@app.get("/health")
def health():
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "ok"}


@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    try:
        contents = await image.read()
        pil_image = Image.open(BytesIO(contents)).convert("RGB")

        preprocess = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.4914, 0.4822, 0.4465],
                std=[0.2470, 0.2435, 0.2616],
            ),
        ])

        tensor = preprocess(pil_image).unsqueeze(0)

        with torch.no_grad():
            probabilities = torch.softmax(model(tensor), dim=1)[0]

        result = {
            CLASS_NAMES[i]: round(float(probabilities[i]), 6)
            for i in range(len(CLASS_NAMES))
        }
        prediction = CLASS_NAMES[int(probabilities.argmax())]

        return {
            "prediction": prediction,
            "probabilities": result,
        }

    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid image: {exc}") from exc
