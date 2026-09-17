import pandas as pd

from src.preprocessing import preprocess_text
from src.ngram_model import train_ngram_model, generate_text


texts = [
    "introduction to data science",
    "introduction to data analysis",
    "data science is useful",
]

df = pd.DataFrame({"text": texts})

df["tokenized_text"] = df["text"].apply(preprocess_text)

model = train_ngram_model(df, n=3)

generated = generate_text(
    seed_text="introduction to",
    n=3,
    model=model,
    max_len=10,
)

print("Tokenized data:")
print(df["tokenized_text"])

print("\nGenerated text:")
print(generated)