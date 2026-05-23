"""Spam Detection App — Streamlit frontend."""

import streamlit as st
import joblib
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from pathlib import Path

# ---------------------------------------------------------------------------
# NLTK data (downloaded once, silently)
# ---------------------------------------------------------------------------
nltk.download("punkt_tab", quiet=True)
nltk.download("stopwords", quiet=True)

# ---------------------------------------------------------------------------
# Module-level singletons (created once, reused across requests)
# ---------------------------------------------------------------------------
_stemmer = PorterStemmer()
_stop_words = frozenset(stopwords.words("english"))

_MODEL_PATH = Path(__file__).parent / "models" / "spam_classifier_model.pkl"
_TFIDF_PATH = Path(__file__).parent / "models" / "tfidf_vectorizer.pkl"


# ---------------------------------------------------------------------------
# Model loading (cached by Streamlit so it's only done once per session)
# ---------------------------------------------------------------------------
@st.cache_resource
def _load_artifacts():
    """Load and cache the trained model and TF-IDF vectorizer."""
    if not _MODEL_PATH.exists() or not _TFIDF_PATH.exists():
        st.error(
            "Model files not found. "
            "Run the training notebook (`notebooks/spam_detection.ipynb`) first."
        )
        st.stop()
    return joblib.load(_MODEL_PATH), joblib.load(_TFIDF_PATH)


# ---------------------------------------------------------------------------
# Text preprocessing (must match the training pipeline exactly)
# ---------------------------------------------------------------------------
def preprocess(text: str) -> str:
    """Lowercase → tokenize → keep alphanumeric → drop stopwords → stem."""
    tokens = nltk.word_tokenize(text.lower())
    return " ".join(
        _stemmer.stem(word)
        for word in tokens
        if word.isalnum() and word not in _stop_words
    )


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------
def main():
    st.set_page_config(page_title="Spam Detector", page_icon="🛡️")
    st.title("🛡️ Spam Detector")
    st.caption("Paste a message below to check if it's spam.")

    model, vectorizer = _load_artifacts()

    message = st.text_area("Message", placeholder="Type or paste a message…")

    if st.button("Classify", type="primary"):
        if not message.strip():
            st.warning("Please enter a message first.")
            return

        processed = preprocess(message)
        prediction = model.predict(vectorizer.transform([processed]))[0]

        if prediction == 1:
            st.error("🚫 **Spam** detected.")
        else:
            st.success("✅ **Not spam** — this message looks safe.")


if __name__ == "__main__":
    main()
