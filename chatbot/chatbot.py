
"""
chatbot.py
-------------------------------------
Semantic Search Chatbot
"""

import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


import joblib
import pandas as pd

from embeddings.bert_embedding import BertEmbedding
from sklearn.metrics.pairwise import cosine_similarity


class CustomerFeedbackChatbot:

    def __init__(self, embedding_file="models/review_embeddings.pkl"):

        print("Loading Chatbot...")

        # Load BERT embedding model
        self.embedder = BertEmbedding()

        # Load customer reviews
        self.df = pd.read_csv("data/reviews.csv")

        # Embedding file path
        self.embedding_file = embedding_file

        # Create models folder if it doesn't exist
        os.makedirs("models", exist_ok=True)

        # Check if embeddings already exist
        if os.path.exists(self.embedding_file):

            print("Loading saved embeddings...")

            self.review_embeddings = joblib.load(self.embedding_file)

            print("Embeddings loaded successfully!")

        else:

            print("No saved embeddings found.")
            print("Generating review embeddings...")

            self.review_embeddings = []

            for review in self.df["review"]:

                embedding = self.embedder.encode(review)

                self.review_embeddings.append(embedding)

            # Save embeddings
            joblib.dump(
                self.review_embeddings,
                self.embedding_file
            )

            print(f"Embeddings saved to: {self.embedding_file}")

        print("Chatbot Ready!")

    ########################################################

    def search(self, question, top_k=5):

        # Generate embedding for user question
        question_embedding = self.embedder.encode(question)

        # Calculate cosine similarity
        scores = cosine_similarity(
            [question_embedding],
            self.review_embeddings
        )[0]

        # Add similarity score to dataframe
        result_df = self.df.copy()

        result_df["score"] = scores

        # Return Top-K similar reviews
        result = result_df.sort_values(
            by="score",
            ascending=False
        ).head(top_k)

        return result[["review", "score"]]


############################################################

if __name__ == "__main__":

    chatbot = CustomerFeedbackChatbot()

    print("=" * 60)

    while True:

        question = input("\nAsk Question : ")

        if question.lower() == "exit":
            break

        result = chatbot.search(question)

        print("\nMost Relevant Reviews\n")

        print(result)

        print("=" * 60)