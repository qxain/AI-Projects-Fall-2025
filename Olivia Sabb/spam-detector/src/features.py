
from sklearn.feature_extraction.text import TfidfVectorizer

def build_vectorizer(kind="word", max_features=50000):
    if kind == "word":
        return TfidfVectorizer(
            ngram_range=(1,2),
            stop_words='english',
            max_features=max_features,
            sublinear_tf=True
        )
    elif kind == "char":
        return TfidfVectorizer(
            analyzer='char',
            ngram_range=(3,5),
            max_features=max_features,
            sublinear_tf=True
        )
    else:
        raise ValueError("kind must be 'word' or 'char'")