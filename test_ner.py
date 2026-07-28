import pandas as pd

from ner.ner_model import NamedEntityRecognizer

ner = NamedEntityRecognizer()

df = pd.read_csv("data/reviews.csv")

result = ner.process_dataframe(df)

print(result)

result.to_csv(
    "data/reviews_with_entities.csv",
    index=False
)

print("\nSaved Successfully!")