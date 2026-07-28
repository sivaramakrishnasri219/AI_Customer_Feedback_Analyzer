import pandas as pd

from sentiment.sentiment_model import SentimentAnalyzer

analyzer = SentimentAnalyzer()

df = pd.read_csv("data/reviews.csv")

result = analyzer.predict_dataframe(df)

print(result)

result.to_csv(
    "data/reviews_with_sentiment.csv",
    index=False
)

print("\nSaved Successfully!")