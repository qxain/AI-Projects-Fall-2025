import pandas as pd
from pathlib import Path
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report
from joblib import dump
from features import build_vectorizer

def load_split(csv_path: str):
    df = pd.read_csv(csv_path)
    return df['text'].astype(str).tolist(), df['label'].astype(int).tolist()

def gridsearch_lr(train_csv, out_dir="experiments"):
    X_train, y_train = load_split(train_csv)
    pipe = Pipeline([
        ("tfidf", build_vectorizer("word")),
        ("clf", LogisticRegression(max_iter=1000, solver='liblinear')),
    ])
    param_grid = {
        "clf__C": [0.5, 1.0, 2.0, 5.0],
        "tfidf__ngram_range": [(1,1), (1,2)],
        "tfidf__max_features": [30000, 50000, 100000],
    }
    gs = GridSearchCV(pipe, param_grid, cv=3, n_jobs=-1, scoring='f1')
    gs.fit(X_train, y_train)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(out_dir) / "best_lr_params.txt", "w") as f:
        f.write(str(gs.best_params_))
        f.write(f"\nBest F1: {gs.best_score_:.4f}\n")
    dump(gs.best_estimator_, Path(out_dir) / "best_lr_model.joblib")
    print("Best:", gs.best_params_, gs.best_score_)

def gridsearch_svm(train_csv, out_dir="experiments"):
    X_train, y_train = load_split(train_csv)
    pipe = Pipeline([
        ("tfidf", build_vectorizer("char")),
        ("clf", CalibratedClassifierCV(LinearSVC(), cv=3)),
    ])
    param_grid = {
        "clf__base_estimator__C": [0.5, 1.0, 2.0],
        "tfidf__ngram_range": [(3,5), (3,6)],
        "tfidf__max_features": [50000, 100000],
    }
    gs = GridSearchCV(pipe, param_grid, cv=3, n_jobs=-1, scoring='f1')
    gs.fit(X_train, y_train)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(out_dir) / "best_svm_params.txt", "w") as f:
        f.write(str(gs.best_params_))
        f.write(f"\nBest F1: {gs.best_score_:.4f}\n")
    dump(gs.best_estimator_, Path(out_dir) / "best_svm_model.joblib")
    print("Best:", gs.best_params_, gs.best_score_)