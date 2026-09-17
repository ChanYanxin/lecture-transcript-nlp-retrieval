import pandas as pd

from src.preprocessing import preprocess_text
from src.retrieval import hierarchical_search


data = [
    {
        "lecture": "01-introduction",
        "start": 0.0,
        "end": 30.0,
        "text": "Welcome to the introduction to data science.",
    },
    {
        "lecture": "01-introduction",
        "start": 30.0,
        "end": 60.0,
        "text": "Today we discuss machine learning and data analysis.",
    },
    {
        "lecture": "10-frequent-itemsets",
        "start": 0.0,
        "end": 30.0,
        "text": "Customers often buy beer and diapers together.",
    },
    {
        "lecture": "10-frequent-itemsets",
        "start": 30.0,
        "end": 60.0,
        "text": "Frequent itemsets describe items purchased together.",
    },
]


df = pd.DataFrame(data)

df["tokenized_text"] = (
    df["text"].apply(preprocess_text)
)


results = hierarchical_search(
    df=df,
    query="beer and diapers",
    top_k=2,
    top_m=2,
)


for result in results:
    print(
        f"\nLecture: {result['lecture']} "
        f"(score={result['lecture_score']:.4f})"
    )

    for segment in result["segments"]:
        print(
            f"  [{segment['start']:.0f}-{segment['end']:.0f}s] "
            f"score={segment['score']:.4f}"
        )
        print(
            f"  {segment['text']}"
        )