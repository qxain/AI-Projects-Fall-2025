
import pandas as pd
import joblib
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from features import build_vectorizer
from models import build_model

def load_split(csv_path: str):
    df = pd.read_csv(csv_path)
    return df['text'].astype(str).tolist(), df['label'].astype(int).tolist()

def train_and_evaluate(train_csv, test_csv, vec_kind="word", model_name="lr", out_dir="experiments"):
    X_train, y_train = load_split(train_csv)
    X_test, y_test = load_split(test_csv)

    vectorizer = build_vectorizer(kind=vec_kind)
    model = build_model(model_name)

    pipe = Pipeline([
        ("tfidf", vectorizer),
        ("clf", model),
    ])

    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=["ham", "spam"], output_dict=False)
    cm = confusion_matrix(y_test, y_pred)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, out / f"model_{model_name}_{vec_kind}.joblib")

    (out / "reports").mkdir(exist_ok=True)
    with open(out / "reports" / f"report_{model_name}_{vec_kind}.txt", "w") as f:
        f.write(f"Accuracy: {acc:.4f}\n\n")
        f.write("Classification report:\n")
        f.write(report)
        f.write("\nConfusion Matrix:\n")
        f.write(str(cm))

    print(f"[{model_name}/{vec_kind}] Accuracy: {acc:.4f}")
    print(report)
    print("Confusion Matrix:")
    print(cm)

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--train_csv", default="data/processed/train.csv")
    p.add_argument("--test_csv", default="data/processed/test.csv")
    p.add_argument("--vec_kind", choices=["word", "char"], default="word")
    p.add_argument("--model_name", choices=["nb", "lr", "svm"], default="lr")
    p.add_argument("--out_dir", default="experiments")
    args = p.parse_args()
    train_and_evaluate(args.train_csv, args.test_csv, args.vec_kind, args.model_name, args.out_dir)