"""
ner_model.py
---------------------------------
Named Entity Recognition using spaCy
"""

import spacy
import pandas as pd

# Load spaCy English Model
nlp = spacy.load("en_core_web_sm")


class NamedEntityRecognizer:

    def __init__(self):

        print("Loading spaCy NER Model...")

        self.nlp = nlp

        print("Model Loaded Successfully!")

    ###########################################################

    def extract_entities(self, text):

        """
        Extract named entities from text.

        Returns:
            List of dictionaries
        """

        doc = self.nlp(text)

        entities = []

        for ent in doc.ents:

            entities.append({

                "text": ent.text,

                "label": ent.label_

            })

        return entities

    ###########################################################

    def print_entities(self, text):

        entities = self.extract_entities(text)

        print("\nReview")

        print(text)

        print("\nEntities")

        if not entities:
            print("No entities found.")
        else:
            for entity in entities:
                print(
                    f"{entity['text']} --> {entity['label']}"
                )

    ###########################################################

    def process_dataframe(
            self,
            dataframe,
            text_column="review"
    ):

        all_entities = []

        for review in dataframe[text_column]:

            entities = self.extract_entities(review)

            entity_text = ", ".join(
                [
                    f"{e['text']} ({e['label']})"
                    for e in entities
                ]
            )

            all_entities.append(entity_text)

        dataframe["entities"] = all_entities

        return dataframe


##############################################################

if __name__ == "__main__":

    ner = NamedEntityRecognizer()

    reviews = [

        "John bought an iPhone 16 from Amazon yesterday.",

        "Google opened a new office in Hyderabad.",

        "Samsung Galaxy S25 costs ₹65000.",

        "Alice visited Delhi last week."

    ]

    print("=" * 70)

    for review in reviews:

        ner.print_entities(review)

        print("-" * 70)