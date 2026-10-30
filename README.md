# IMDB Sentiment Analysis

Classifies movie reviews as positive or negative, trained on the IMDB dataset (50,000 reviews).

## Folder structure

```
sentiment-analysis/
├── index.html         # web page: paste a review, get the sentiment
├── train_model.py     # trains the model on IMDB and writes model.js
├── model.js           # created by train_model.py (word weights + accuracy)
├── requirements.txt
└── README.md
```

## How to run

1. `pip install -r requirements.txt`
2. `python train_model.py`   (downloads IMDB, trains, creates model.js)
3. Double-click `index.html` to open it in your browser.

Without step 2, index.html still opens in a simple demo mode that uses a small word list.

## How it works

- TF-IDF turns each review into numbers (8,000 most useful words).
- Logistic regression learns a weight per word: positive weights for words like "superb", negative for "waste".
- The page repeats the same TF-IDF calculation in JavaScript and shows the words that influenced the result.
