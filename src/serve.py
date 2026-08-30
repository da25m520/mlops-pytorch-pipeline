import os

import torch
from fastapi import FastAPI, File, UploadFile, HTTPException
from PIL import Image
from torchvision import transforms

from src.model import get_model


app = FastAPI(title="PyTorch Model API")


# --------------------------------------------------
# Configuration
# --------------------------------------------------

MODEL_PATH = os.getenv(
    "MODEL_PATH",
    os.path.join("checkpoints", "classifier_v1.pt"),
)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# --------------------------------------------------
# Load model checkpoint
# --------------------------------------------------

model = None
model_loaded = False

try:
    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
    )

    architecture = checkpoint.get(
        "architecture",
        "simple_cnn",
    )

    num_classes = checkpoint.get(
        "num_classes",
        10,
    )

    model = get_model(
        architecture=architecture,
        num_classes=num_classes,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.to(DEVICE)
    model.eval()

    model_loaded = True

except Exception as exc:
    print(
        f"Failed to load model from {MODEL_PATH}: {exc}"
    )


# --------------------------------------------------
# CIFAR-10 preprocessing
# --------------------------------------------------

preprocess = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2470, 0.2435, 0.2616],
    ),
])


# --------------------------------------------------
# Health endpoint
# --------------------------------------------------

@app.get("/health")
def health_check():
    if not model_loaded:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded",
        )

    return {
        "status": "ok",
        "model_loaded": True,
    }


# --------------------------------------------------
# Prediction endpoint
# --------------------------------------------------

@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    if not model_loaded:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded",
        )

    try:
        image_bytes = await image.read()

        pil_image = Image.open(
            __import__("io").BytesIO(image_bytes)
        ).convert("RGB")

        x = preprocess(pil_image)
        x = x.unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            outputs = model(x)
            probabilities = torch.softmax(
                outputs,
                dim=1,
            )[0]

        predicted_class = torch.argmax(
            probabilities
        ).item()

        return {
            "prediction": predicted_class,
            "probabilities": [
                round(float(prob), 6)
                for prob in probabilities
            ],
        }

    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid image: {exc}",
        )