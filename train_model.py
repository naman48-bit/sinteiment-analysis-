"""
Train a sentiment classifier on the IMDB movie review dataset (50,000 reviews)
and export it as model.js so index.html can use it directly in the browser.

Run:
    pip install -r requirements.txt
    python train_model.py
"""
import json
import re

from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def clean(text):
    # IMDB reviews contain "<br />" tags. index.html applies the same rule.
    return re.sub(r"<br\s*/?>", " ", text.lower())


print("Loading IMDB dataset...")
data = load_dataset("imdb")
x_train, y_train = data["train"]["text"], data["train"]["label"]
x_test, y_test = data["test"]["text"], data["test"]["label"]

print("Building TF-IDF features...")
vectorizer = TfidfVectorizer(
    preprocessor=clean,
    token_pattern=r"[a-z']+",
    max_features=8000,
    min_df=5,
    sublinear_tf=True,
)
X_train = vectorizer.fit_transform(x_train)
X_test = vectorizer.transform(x_test)

print("Training logistic regression...")
model = LogisticRegression(C=4.0, max_iter=1000)
model.fit(X_train, y_train)

pred = model.predict(X_test)
accuracy = accuracy_score(y_test, pred)
print(f"\nTest accuracy: {accuracy:.4f}\n")
print(classification_report(y_test, pred, target_names=["negative", "positive"]))
print("Confusion matrix [[TN, FP], [FN, TP]]:")
print(confusion_matrix(y_test, pred))

# Export everything the browser needs: per-word idf + weight, and the bias.
idf = vectorizer.idf_
weights = model.coef_[0]
vocab = {
    word: [round(float(idf[i]), 4), round(float(weights[i]), 4)]
    for word, i in vectorizer.vocabulary_.items()
}
export = {
    "vocab": vocab,
    "bias": round(float(model.intercept_[0]), 4),
    "accuracy": round(float(accuracy), 4),
    "features": len(vocab),
    "train_size": len(x_train),
    "test_size": len(x_test),
}

with open("model.js", "w", encoding="utf-8") as f:
    f.write("window.SENTIMENT_MODEL = " + json.dumps(export) + ";\n")

print("\nSaved model.js. Open index.html to try it.")
