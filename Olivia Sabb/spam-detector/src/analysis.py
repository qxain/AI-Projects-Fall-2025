import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

def top_features_from_lr(pipeline, top_k=30):
    vec = pipeline.named_steps['tfidf']
    clf = pipeline.named_steps['clf']
    if not hasattr(clf, "coef_"):
        raise ValueError("Classifier must expose coef_ (e.g., LogisticRegression).")
    feature_names = np.array(vec.get_feature_names_out())
    coefs = clf.coef_[0]
    top_spam_idx = np.argsort(coefs)[-top_k:][::-1]
    top_ham_idx = np.argsort(coefs)[:top_k]
    spam = list(zip(feature_names[top_spam_idx], coefs[top_spam_idx]))
    ham = list(zip(feature_names[top_ham_idx], coefs[top_ham_idx]))
    return {"spam": spam, "ham": ham}

def plot_confusion_matrix(y_true, y_pred, out_path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(4,3))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['ham','spam'], yticklabels=['ham','spam'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

def print_report(y_true, y_pred):
    print(classification_report(y_true, y_pred, target_names=["ham", "spam"]))

def load_pipeline(model_path):
    return joblib.load(model_path)