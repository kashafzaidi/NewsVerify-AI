import pickle
import re


# ==========================================
# STEP 1: LOAD TRAINED MODEL
# ==========================================

with open("model/fake_news_model.pkl", "rb") as file:
    model = pickle.load(file)


# ==========================================
# STEP 2: LOAD TF-IDF VECTORIZER
# ==========================================

with open("model/tfidf_vectorizer.pkl", "rb") as file:
    vectorizer = pickle.load(file)


# ==========================================
# STEP 3: TEXT CLEANING FUNCTION
# ==========================================

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text


# ==========================================
# STEP 4: PREDICTION FUNCTION
# ==========================================

def predict_news(news):

    # Clean news
    cleaned_news = clean_text(news)

    # Convert text into TF-IDF
    news_tfidf = vectorizer.transform([cleaned_news])

    # Predict
    prediction = model.predict(news_tfidf)[0]

    # Result
    if prediction == 0:
        return "FAKE NEWS"
    else:
        return "REAL NEWS"


# ==========================================
# STEP 5: TEST THE MODEL
# ==========================================

def predict_news(news):

    cleaned_news = clean_text(news)

    news_tfidf = vectorizer.transform([cleaned_news])

    prediction = model.predict(news_tfidf)[0]

    if prediction == 0:
        return "FAKE NEWS"
    else:
        return "REAL NEWS"