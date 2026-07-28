import pandas as pd
from preprocessing.clean_text import TextCleaner
cleaner = TextCleaner()
df = pd.read_csv("data/reviews.csv")
print("Original Reviews\n")
print(df.head())
print("\nCleaned Reviews\n")
df["clean_review"] = df["review"].apply(cleaner.clean)
print(df.head())