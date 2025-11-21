from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV

def build_model(name: str):
    name = name.lower()
    if name == "nb":
        return MultinomialNB(alpha=0.1)
    elif name == "lr":
        return LogisticRegression(max_iter=500, solver='liblinear', class_weight=None)
    elif name == "svm":
        svm = LinearSVC(C=1.0)
        # Calibrate to get probabilities for metrics like ROC-AUC if needed
        return CalibratedClassifierCV(svm, cv=3)
    else:
        raise ValueError("Unknown model name")