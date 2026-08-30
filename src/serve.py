import torch
from fastapi import FastAPI
from pydantic import BaseModel

from src.model import SimpleClassifier


app = FastAPI(title="PyTorch Model API")


# Create/load model
model = SimpleClassifier(
    input_size=4,
    hidden_size=16,
    num_classes=3
)

model.eval()


class PredictionRequest(BaseModel):
    features: list[float]


@app.get("/")
def health_check():
    return {"status": "ok"}


@app.post("/predict")
def predict(request: PredictionRequest):
    x = torch.tensor([request.features], dtype=torch.float32)

    with torch.no_grad():
        output = model(x)
        prediction = torch.argmax(output, dim=1).item()

    return {
        "prediction": prediction
    }