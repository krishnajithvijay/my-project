import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')

# Load new dataset model & vectorizer
with open("fake_news_model_new.pkl", "rb") as f:
    model = pickle.load(f)

with open("tfidf_vectorizer_new.pkl", "rb") as f:
    tfidf = pickle.load(f)

stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return " ".join(words)

st.title("Fake News Detection")
st.write("Model trained on content-level labeled dataset")

user_input = st.text_area("Enter news text")

if st.button("Check News"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        cleaned = clean_text(user_input)
        vector = tfidf.transform([cleaned])
        prediction = model.predict(vector)

        if prediction[0] == 0:
            st.error("🚨 FAKE NEWS")
        else:
            st.success("✅ REAL NEWS")
