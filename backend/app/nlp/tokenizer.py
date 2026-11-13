import re
from collections import Counter
from typing import List
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Initialize NLTK safely
for pkg in ["punkt", "punkt_tab", "stopwords", "wordnet"]:
    try:
        nltk.download(pkg, quiet=True)
    except Exception:
        pass

lemmatizer = WordNetLemmatizer()


def clean_and_tokenize(text: str) -> List[str]:
    """
    NLP preprocessing pipeline:
      1. Lowercase
      2. Normalize punctuation (retaining + and # for C++, C#)
      3. Tokenize
      4. Remove standard stopwords
      5. Lemmatize tokens
    """
    text_lower = text.lower()
    text_clean = re.sub(r"[^a-z0-9\s\+\#]", " ", text_lower)
    try:
        tokens = word_tokenize(text_clean)
    except Exception:
        tokens = text_clean.split()

    try:
        stop_words = set(stopwords.words("english"))
    except Exception:
        stop_words = {"the", "and", "is", "in", "to", "of", "for", "with", "a", "an", "on", "at"}

    return [
        lemmatizer.lemmatize(t)
        for t in tokens
        if t not in stop_words and len(t) > 1
    ]


def get_top_keywords(text: str, top_n: int = 15) -> List[str]:
    """Extract top frequent informative keywords from text."""
    tokens = clean_and_tokenize(text)
    freq = Counter(t for t in tokens if len(t) > 2)
    return [w for w, _ in freq.most_common(top_n)]
