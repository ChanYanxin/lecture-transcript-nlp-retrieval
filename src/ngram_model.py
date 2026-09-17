import random

from nltk import ConditionalFreqDist
from nltk.util import ngrams

from .preprocessing import preprocess_text


def train_ngram_model(df, n):
    """
    Train an N-gram language model from tokenized transcript segments.

    Parameters
    ----------
    df : pandas.DataFrame
        Must contain a 'tokenized_text' column.
    n : int
        Order of the N-gram model.

    Returns
    -------
    ConditionalFreqDist
        Maps each (n-1)-token context to a frequency distribution
        of possible next tokens.
    """
    if n < 2:
        raise ValueError("n must be at least 2.")

    all_ngrams = []

    for tokens in df["tokenized_text"]:
        if not isinstance(tokens, list):
            continue

        # Add sentence-boundary tokens.
        padded_tokens = (
            ["<s>"] * (n - 1)
            + tokens
            + ["</s>"]
        )

        all_ngrams.extend(
            ngrams(padded_tokens, n)
        )

    # Convert:
    #
    # ("data", "science", "is")
    #
    # into:
    #
    # context = ("data", "science")
    # next_word = "is"
    cfd_input = [
        (tuple(ngram[:-1]), ngram[-1])
        for ngram in all_ngrams
    ]

    return ConditionalFreqDist(cfd_input)


def predict_next_word(context, model, seed=32133):
    """
    Sample one next token given an (n-1)-token context.
    """
    if context not in model:
        return None

    frequency_distribution = model[context]

    # Sorting ensures deterministic candidate ordering.
    candidates = sorted(
        frequency_distribution.keys()
    )

    weights = [
        frequency_distribution[word]
        for word in candidates
    ]

    # Required by the original coursework for reproducibility.
    random.seed(seed)

    return random.choices(
        candidates,
        weights=weights,
        k=1
    )[0]


def generate_text(seed_text, n, model, max_len=30):
    """
    Generate text autoregressively from a seed phrase.
    """
    context_tokens = preprocess_text(seed_text)

    # An N-gram model requires n-1 previous tokens.
    while len(context_tokens) < n - 1:
        context_tokens = ["<s>"] + context_tokens

    result = list(context_tokens)

    for _ in range(max_len):
        context = tuple(
            result[-(n - 1):]
        )

        next_word = predict_next_word(
            context,
            model
        )

        if next_word is None or next_word == "</s>":
            break

        result.append(next_word)

    # Do not show artificial start tokens in the final sentence.
    output_tokens = [
        token
        for token in result
        if token != "<s>"
    ]

    return " ".join(output_tokens)