import pandas as pd
import re
import pickle

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# =========================================================
# STEP 1: LOAD DATASET
# =========================================================

fake = pd.read_csv("data/Fake.csv")
true = pd.read_csv("data/True.csv")

print("Fake:", fake.shape)
print("True:", true.shape)


# =========================================================
# STEP 2: CHECK MISSING VALUES
# =========================================================

print("\nFake missing values:")
print(fake.isnull().sum())

print("\nTrue missing values:")
print(true.isnull().sum())


# =========================================================
# STEP 3: ADD LABELS
# =========================================================

# 0 = Fake News
# 1 = Real News

fake["label"] = 0
true["label"] = 1


# =========================================================
# STEP 4: COMBINE BOTH DATASETS
# =========================================================

data = pd.concat([fake, true], axis=0)

print("\nCombined dataset shape:", data.shape)


# =========================================================
# STEP 5: SHUFFLE DATA
# =========================================================

data = data.sample(
    frac=1,
    random_state=42
).reset_index(drop=True)


# =========================================================
# STEP 6: COMBINE TITLE + TEXT
# =========================================================

data["content"] = data["title"] + " " + data["text"]


# =========================================================
# STEP 7: TEXT CLEANING
# =========================================================

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text


data["content"] = data["content"].apply(clean_text)


# =========================================================
# STEP 8: CREATE X AND Y
# =========================================================

X = data["content"]
y = data["label"]


# =========================================================
# STEP 9: TRAIN TEST SPLIT
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining data:", X_train.shape)
print("Testing data:", X_test.shape)


# =========================================================
# STEP 10: TF-IDF VECTORIZATION
# =========================================================

vectorizer = TfidfVectorizer(
    stop_words="english"
)

X_train_tfidf = vectorizer.fit_transform(X_train)

X_test_tfidf = vectorizer.transform(X_test)

print("\nTF-IDF Training shape:", X_train_tfidf.shape)
print("TF-IDF Testing shape:", X_test_tfidf.shape)


# =========================================================
# STEP 11: TRAIN LOGISTIC REGRESSION MODEL
# =========================================================

model = LogisticRegression(
    max_iter=1000
)

model.fit(X_train_tfidf, y_train)

print("\nModel training completed!")


# =========================================================
# STEP 12: PREDICTION
# =========================================================

y_pred = model.predict(X_test_tfidf)


# =========================================================
# STEP 13: MODEL EVALUATION
# =========================================================

accuracy = accuracy_score(y_test, y_pred)

print("\n==============================")
print("MODEL PERFORMANCE")
print("==============================")

print("\nAccuracy:", accuracy)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))


# =========================================================
# STEP 14: SAVE MODEL
# =========================================================

with open("model/fake_news_model.pkl", "wb") as file:
    pickle.dump(model, file)

print("\nML Model saved successfully!")


# =========================================================
# STEP 15: SAVE TF-IDF VECTORIZER
# =========================================================

with open("model/tfidf_vectorizer.pkl", "wb") as file:
    pickle.dump(vectorizer, file)

print("TF-IDF Vectorizer saved successfully!")


# =========================================================
# FINAL MESSAGE
# =========================================================

print("\n===================================")
print("FAKE NEWS DETECTION MODEL READY!")
print("===================================")