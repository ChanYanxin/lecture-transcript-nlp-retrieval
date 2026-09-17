import string

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer


stop_words = set(stopwords.words("english"))
stemmer = SnowballStemmer("english")


def preprocess_text(text):
    """
    Basic text preprocessing.

    Steps:
    1. Convert text to lowercase.
    2. Remove punctuation.
    3. Tokenize text.

    Stopwords are intentionally retained because this preprocessing
    is also used for N-gram language modeling.
    """
    text = str(text).lower()

    text = text.translate(
        str.maketrans("", "", string.punctuation)
    )

    tokens = word_tokenize(text)

    return tokens


def retrieval_analyzer(text):
    """
    Additional preprocessing for TF-IDF retrieval.

    Steps:
    1. Apply basic preprocessing.
    2. Remove stopwords.
    3. Apply stemming.
    """
    tokens = preprocess_text(text)

    processed_tokens = [
        stemmer.stem(token)
        for token in tokens
        if token not in stop_words
    ]

    return processed_tokens