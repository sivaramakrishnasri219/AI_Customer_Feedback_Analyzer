"""
sentiment_model.py
--------------------------------
Sentiment Analysis using Hugging Face Transformers
"""

from transformers import pipeline
import pandas as pd


class SentimentAnalyzer:
    def __init__(self):
        print("Loading Sentiment Model...")

        self.classifier = pipeline(
            task="sentiment-analysis",
            model="cardiffnlp/twitter-roberta-base-sentiment-latest"
        )

        print("Sentiment Model Loaded Successfully!")

    ###########################################################

    def predict(self, text):

        result = self.classifier(text)[0]

        sentiment = result["label"]
        confidence = round(result["score"], 4)

        return {
            "text": text,
            "sentiment": sentiment,
            "confidence": confidence
        }

    ###########################################################

    def predict_dataframe(
            self,
            dataframe,
            text_column="review"
    ):

        sentiments = []
        confidences = []

        for review in dataframe[text_column]:
            result = self.predict(review)
            sentiments.append(result["sentiment"])
            confidences.append(result["confidence"])
        dataframe["sentiment"] = sentiments
        dataframe["confidence"] = confidences
        return dataframe


##############################################################

if __name__ == "__main__":
    analyzer = SentimentAnalyzer()
    reviews = [
        "I love this phone. Amazing camera.",
        "Battery backup is terrible.",
        "Delivery was okay.",
        "Worst customer support ever.",
        "Excellent performance and display."

    ]
    print("=" * 70)
    for review in reviews:
        result = analyzer.predict(review)
        print(result)
        print("-" * 70)