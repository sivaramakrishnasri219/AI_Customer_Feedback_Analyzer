"""
bert_embedding.py
-----------------------------------
Generate sentence embeddings using BERT
"""

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


class BertEmbedding:

    def __init__(self):

        print("Loading BERT model...")

        self.model = SentenceTransformer(
            "all-MiniLM-L6-v2"
        )

        print("Model Loaded Successfully!")

    ########################################################

    def encode(self, text):

        embedding = self.model.encode(text)

        return embedding

    ########################################################

    def similarity(self, sentence1, sentence2):

        emb1 = self.encode(sentence1)

        emb2 = self.encode(sentence2)

        score = cosine_similarity(
            [emb1],
            [emb2]
        )[0][0]

        return score


############################################################

if __name__ == "__main__":

    bert = BertEmbedding()

    sentence1 = "I love this phone."

    sentence2 = "This mobile is amazing."

    sentence3 = "The weather is very hot today."

    print("\nSentence 1 Embedding Shape")

    emb = bert.encode(sentence1)

    print(emb.shape)

    print("\nSimilarity")

    print(
        sentence1,
        "<->",
        sentence2
    )

    score = bert.similarity(
        sentence1,
        sentence2
    )

    print(score)

    print("\nSimilarity")

    print(
        sentence1,
        "<->",
        sentence3
    )

    score = bert.similarity(
        sentence1,
        sentence3
    )

    print(score)