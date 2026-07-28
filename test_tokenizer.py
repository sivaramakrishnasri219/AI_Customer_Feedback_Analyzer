import pandas as pd

from preprocessing.clean_text import TextCleaner
from preprocessing.tokenizer import TextTokenizer

cleaner = TextCleaner()
tokenizer = TextTokenizer()

df = pd.read_csv("data/reviews.csv")

for review in df["review"]:

    print("=" * 60)

    print("Original")

    print(review)

    clean = cleaner.clean(review)

    print("\nClean")

    print(clean)

    tokens = tokenizer.nltk_tokenize(clean)

    print("\nTokens")

    print(tokens)

    filtered = tokenizer.remove_stopwords(tokens)

    print("\nWithout Stopwords")

    print(filtered)