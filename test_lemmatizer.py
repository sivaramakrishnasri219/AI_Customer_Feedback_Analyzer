import pandas as pd
from preprocessing.clean_text import TextCleaner
from preprocessing.tokenizer import TextTokenizer
from preprocessing.lemmatizer import TextLemmatizer
cleaner = TextCleaner()
tokenizer = TextTokenizer()
lemmatizer = TextLemmatizer()
df = pd.read_csv("data/reviews.csv")
for review in df["review"]:
    print("=" * 60)
    print("Original")
    print(review)
    clean = cleaner.clean(review)
    print("\nClean")
    print(clean)
    tokens = tokenizer.nltk_tokenize(clean)
    tokens = tokenizer.remove_stopwords(tokens)
    print("\nTokens")
    print(tokens)
    lemmas = lemmatizer.nltk_lemmatize(tokens)
    print("\nNLTK Lemmas")
    print(lemmas)
    spacy_lemmas = lemmatizer.spacy_lemmatize(clean)
    print("\nspaCy Lemmas")
    print(spacy_lemmas)