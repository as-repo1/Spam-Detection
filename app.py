import streamlit as st
import joblib  # Import joblib to load model and vectorizer
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import string

# Function to preprocess text
def transform_texts(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y.copy()
    y.clear()

    ps = PorterStemmer()
    for i in text:
        if i not in stopwords.words("english") and i not in string.punctuation:
            y.append(ps.stem(i))

    text = y.copy()
    y.clear()

    return " ".join(text)

# Load model and TF-IDF vectorizer
model = joblib.load("spam_classifier_model.pkl")
tfidf_vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Streamlit main function
def main():
    st.title("Spam Detection App")

    user_input = st.text_area("Enter a message:")
    if st.button("Predict"):
        processed_input = transform_texts(user_input)
        vectorized_input = tfidf_vectorizer.transform([processed_input])
        prediction = model.predict(vectorized_input)[0]

        if prediction == 1:
            st.error("This message is Spam!")
        else:
            st.success("This message is Ham (Not Spam).")

if __name__ == "__main__":
    main()
