"""
tokenizer.py
-------------------
Tokenize text using NLTK and spaCy
"""

import nltk
import spacy

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Load spaCy English model
nlp = spacy.load("en_core_web_sm")


class TextTokenizer:

    def __init__(self):

        self.stop_words = set(stopwords.words("english"))

    #######################################################
    # NLTK Tokenizer
    #######################################################

    def nltk_tokenize(self, text):

        tokens = word_tokenize(text)

        return tokens

    #######################################################
    # Remove Stopwords
    #######################################################

    def remove_stopwords(self, tokens):

        filtered = [
            word
            for word in tokens
            if word.lower() not in self.stop_words
        ]

        return filtered

    #######################################################
    # spaCy Tokenizer
    #######################################################

    def spacy_tokenize(self, text):

        doc = nlp(text)

        tokens = [token.text for token in doc]

        return tokens


if __name__ == "__main__":

    tokenizer = TextTokenizer()

    sentence = "I absolutely love Python, because it is very easy to learn."

    print("=" * 60)

    print("Original Sentence\n")

    print(sentence)

    print("\n")

    ####################################################

    nltk_tokens = tokenizer.nltk_tokenize(sentence)

    print("NLTK Tokens\n")

    print(nltk_tokens)

    print("\n")

    ####################################################

    filtered = tokenizer.remove_stopwords(nltk_tokens)

    print("After Stopword Removal\n")

    print(filtered)

    print("\n")

    ####################################################

    spacy_tokens = tokenizer.spacy_tokenize(sentence)

    print("spaCy Tokens\n")

    print(spacy_tokens)

    print("=" * 60)