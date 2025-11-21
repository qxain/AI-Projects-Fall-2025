
import pandas as pd
import joblib
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

def evaluate(model_path, test_csv):
    pipe = joblib.load(model_path)
    df = pd.read_csv(test_csv)
    X_test = df['text'].astype(str).tolist()
    y_test = df['label'].astype(int).tolist()
    y_pred = pipe.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred, target_names=["ham", "spam"]))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True)
    p.add_argument("--test_csv", default="data/processed/test.csv")
    args = p.parse_args()
    evaluate(args.model_path, args.test_csv)