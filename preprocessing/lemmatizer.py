"""
lemmatizer.py
--------------------------
Lemmatization using NLTK and spaCy
"""

import spacy

from nltk.stem import WordNetLemmatizer

# Load spaCy English model
nlp = spacy.load("en_core_web_sm")


class TextLemmatizer:

    def __init__(self):

        self.wordnet = WordNetLemmatizer()

    #####################################################
    # NLTK Lemmatizer
    #####################################################

    def nltk_lemmatize(self, tokens):

        lemmas = [
            self.wordnet.lemmatize(word,pos='v')  # Lemmatize as verb
            for word in tokens
        ]

        return lemmas

    #####################################################
    # spaCy Lemmatizer
    #####################################################

    def spacy_lemmatize(self, text):

        doc = nlp(text)

        lemmas = [
            token.lemma_
            for token in doc
            if not token.is_punct
        ]

        return lemmas


if __name__ == "__main__":

    lemmatizer = TextLemmatizer()

    sentence = "The children are playing with better cars."

    tokens = sentence.split()

    print("=" * 60)

    print("Original Tokens")

    print(tokens)

    print("\n")

    nltk_output = lemmatizer.nltk_lemmatize(tokens)

    print("NLTK Lemmatization")

    print(nltk_output)

    print("\n")

    spacy_output = lemmatizer.spacy_lemmatize(sentence)

    print("spaCy Lemmatization")

    print(spacy_output)

    print("=" * 60)