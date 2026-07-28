import pandas as pd

from ner.ner_model import NamedEntityRecognizer
from preprocessing.clean_text import TextCleaner
from preprocessing.tokenizer import TextTokenizer
from preprocessing.lemmatizer import TextLemmatizer
from sentiment.sentiment_model import SentimentAnalyzer

cleaner = TextCleaner()
tokenizer = TextTokenizer()
lemmatizer = TextLemmatizer()
sentiment = SentimentAnalyzer()
ner = NamedEntityRecognizer()

df = pd.read_csv("data/reviews.csv")
for review in df["review"]:

    print("=" * 70)

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

    print("\nLemmas: ")
    print(lemmas)

    result = sentiment.predict(clean)
    print("\nSentiment: ")
    print(result)

    entity_result = ner.extract_entities(review)
    print("\nNamed Entities:")
    print(entity_result)

    