import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer

from .preprocessing import retrieval_analyzer


def build_lecture_index(df):
    """
    Build a TF-IDF index where each lecture is treated as one document.

    Parameters
    ----------
    df : pandas.DataFrame
        Must contain:
        - lecture
        - tokenized_text

    Returns
    -------
    lecture_names : list[str]
        Lecture identifiers.

    lecture_corpus : list[str]
        One combined document per lecture.

    vectorizer : TfidfVectorizer
        Fitted TF-IDF vectorizer.

    tfidf_matrix
        TF-IDF representation of all lectures.
    """

    lecture_groups = (
        df.groupby("lecture")["tokenized_text"]
        .apply(
            lambda segments:
            " ".join(
                token
                for segment in segments
                for token in segment
            )
        )
        .reset_index(name="document")
    )

    lecture_names = lecture_groups["lecture"].tolist()
    lecture_corpus = lecture_groups["document"].tolist()

    vectorizer = TfidfVectorizer(
        analyzer=retrieval_analyzer
    )

    tfidf_matrix = vectorizer.fit_transform(
        lecture_corpus
    )

    return (
        lecture_names,
        lecture_corpus,
        vectorizer,
        tfidf_matrix,
    )


def search_lectures(
    query,
    lecture_names,
    vectorizer,
    tfidf_matrix,
    top_k=2,
):
    """
    Retrieve the top-k lectures most relevant to a query.
    """

    query_vector = vectorizer.transform([query])

    scores = (
        tfidf_matrix @ query_vector.T
    ).toarray().ravel()

    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []

    for index in top_indices:
        results.append(
            {
                "lecture": lecture_names[index],
                "score": float(scores[index]),
            }
        )

    return results


def search_segments(
    df,
    lecture_name,
    query,
    top_m=2,
):
    """
    Search timestamp-level transcript segments inside one lecture.
    """

    lecture_df = (
        df[df["lecture"] == lecture_name]
        .copy()
        .reset_index(drop=True)
    )

    segment_corpus = (
        lecture_df["tokenized_text"]
        .apply(lambda tokens: " ".join(tokens))
        .tolist()
    )

    vectorizer = TfidfVectorizer(
        analyzer=retrieval_analyzer
    )

    tfidf_matrix = vectorizer.fit_transform(
        segment_corpus
    )

    query_vector = vectorizer.transform([query])

    scores = (
        tfidf_matrix @ query_vector.T
    ).toarray().ravel()

    top_indices = np.argsort(scores)[::-1][:top_m]

    results = []

    for index in top_indices:
        row = lecture_df.iloc[index]

        results.append(
            {
                "start": float(row["start"]),
                "end": float(row["end"]),
                "score": float(scores[index]),
                "text": str(row["text"]),
            }
        )

    return results


def hierarchical_search(
    df,
    query,
    top_k=2,
    top_m=2,
):
    """
    Perform two-stage hierarchical retrieval:

    Level 1:
        Retrieve the most relevant lectures.

    Level 2:
        Retrieve the most relevant timestamp segments
        inside each selected lecture.
    """

    (
        lecture_names,
        _,
        vectorizer,
        tfidf_matrix,
    ) = build_lecture_index(df)

    lecture_results = search_lectures(
        query=query,
        lecture_names=lecture_names,
        vectorizer=vectorizer,
        tfidf_matrix=tfidf_matrix,
        top_k=top_k,
    )

    results = []

    for lecture_result in lecture_results:
        lecture_name = lecture_result["lecture"]

        segment_results = search_segments(
            df=df,
            lecture_name=lecture_name,
            query=query,
            top_m=top_m,
        )

        results.append(
            {
                "lecture": lecture_name,
                "lecture_score": lecture_result["score"],
                "segments": segment_results,
            }
        )

    return results