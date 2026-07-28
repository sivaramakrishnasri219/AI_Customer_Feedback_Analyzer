import os
import sys

# Add the project root to Python's module search path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

print("Project Root:", PROJECT_ROOT)
print("Python Path:", sys.path[:3])

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud

from preprocessing.clean_text import TextCleaner
from sentiment.sentiment_model import SentimentAnalyzer
from ner.ner_model import NamedEntityRecognizer
from chatbot.chatbot import CustomerFeedbackChatbot
##########################################################

st.set_page_config(

    page_title="AI Customer Feedback Analyzer",

    layout="wide"

)

###########################################################

st.title("🤖 AI Customer Feedback Analyzer")

st.write("Sentiment Analysis + NER + Text Analytics")

###########################################################

@st.cache_resource
def load_models():

    cleaner = TextCleaner()

    sentiment = SentimentAnalyzer()

    ner = NamedEntityRecognizer()

    return cleaner, sentiment, ner


cleaner, sentiment, ner = load_models()

###########################################################

df = pd.read_csv("data/reviews.csv")

###########################################################

sentiments = []

confidences = []

entities = []

clean_reviews = []

###########################################################

for review in df["review"]:

    clean = cleaner.clean(review)

    clean_reviews.append(clean)

    sentiment_result = sentiment.predict(clean)

    sentiments.append(sentiment_result["sentiment"])

    confidences.append(sentiment_result["confidence"])

    entity = ner.extract_entities(review)

    entities.append(entity)

###########################################################

df["clean_review"] = clean_reviews

df["sentiment"] = sentiments

df["confidence"] = confidences

df["entities"] = entities

###########################################################
# Sidebar
###########################################################

st.sidebar.header("Filters")

selected = st.sidebar.selectbox(

    "Select Sentiment",

    ["All", "positive", "neutral", "negative"]

)

search = st.sidebar.text_input("Search Review")

###########################################################

filtered = df.copy()

if selected != "All":

    filtered = filtered[
        filtered["sentiment"] == selected
    ]

if search:

    filtered = filtered[
        filtered["review"].str.contains(
            search,
            case=False
        )
    ]

###########################################################
# Metrics
###########################################################

st.subheader("Dashboard Metrics")

c1, c2, c3, c4 = st.columns(4)

c1.metric("Total Reviews", len(filtered))

c2.metric(
    "Positive",
    (filtered["sentiment"] == "positive").sum()
)

c3.metric(
    "Negative",
    (filtered["sentiment"] == "negative").sum()
)

c4.metric(
    "Neutral",
    (filtered["sentiment"] == "neutral").sum()
)

###########################################################
# Sentiment Bar Chart
###########################################################

st.subheader("Sentiment Distribution")

count = filtered["sentiment"].value_counts()

fig, ax = plt.subplots()

count.plot(
    kind="bar",
    ax=ax
)

st.pyplot(fig)

###########################################################
# Word Cloud
###########################################################

st.subheader("Word Cloud")

text = " ".join(filtered["clean_review"])

wc = WordCloud(

    width=900,

    height=400,

    background_color="white"

).generate(text)

fig, ax = plt.subplots(figsize=(12,5))

ax.imshow(wc)

ax.axis("off")

st.pyplot(fig)

###########################################################
# Entities
###########################################################

st.subheader("Named Entities")

for i, row in filtered.iterrows():

    st.write("Review")

    st.write(row["review"])

    st.write("Entities")

    st.write(row["entities"])

    st.divider()

###########################################################
# Data Table
###########################################################

st.subheader("Processed Dataset")

st.dataframe(filtered)

############################################################
chatbot = CustomerFeedbackChatbot()

st.subheader("AI Customer Chatbot")
question = st.text_input(
    "Ask about customer reviews"
)
if question:
    answer = chatbot.search(question)
    st.dataframe(answer)