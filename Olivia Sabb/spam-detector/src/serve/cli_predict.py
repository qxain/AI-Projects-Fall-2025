import sys
import joblib
from pathlib import Path

def main():
    if len(sys.argv) < 2:
        print("Usage: python -m src.serve.cli_predict '<your email text here>'")
        sys.exit(1)
    model_path = Path("experiments/model_lr_word.joblib")
    pipe = joblib.load(model_path)
    text = " ".join(sys.argv[1:])
    pred = pipe.predict([text])[0]
    label = "spam" if pred == 1 else "ham"
    proba = getattr(pipe.named_steps['clf'], "predict_proba", None)
    if proba:
        p = proba([text])[0][1]
        print(f"{label} (p_spam={p:.3f})")
    else:
        print(label)

if __name__ == "__main__":
    main()