from fastapi import FastAPI
from pydantic import BaseModel
import joblib
from pathlib import Path

app = FastAPI(title="Spam Detector API", version="1.0")

# Load the model once, when the API starts
BASE_DIR = Path(__file__).resolve().parents[2]

model_path = BASE_DIR / "notebooks" / "experiments" / "model_lr_word.joblib"

pipe = joblib.load(model_path)

class EmailIn(BaseModel):
    text: str

class PredictionOut(BaseModel):
    label: str
    probability_spam: float | None = None


@app.get("/health", tags=["Health"])
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionOut, tags=["Prediction"])
def predict(email: EmailIn):
    """
    Predict whether an email is spam or ham
    """

    pred = pipe.predict([email.text])[0]
    label = "spam" if pred == 1 else "ham"

    # Retrieve probabilities if the model supports it
    probability_spam = None
    if hasattr(pipe, "predict_proba"):
        probability_spam = float(pipe.predict_proba([email.text])[0][1])

    return {"spam_or_not": label, "probability_spam": probability_spam}
