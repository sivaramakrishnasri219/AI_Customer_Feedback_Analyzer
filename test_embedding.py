import pandas as pd

from embeddings.bert_embedding import BertEmbedding

bert = BertEmbedding()

df = pd.read_csv("data/reviews.csv")

print("=" * 70)

for review in df["review"]:

    emb = bert.encode(review)

    print(review)

    print("Embedding Shape :", emb.shape)

    print("-" * 70)