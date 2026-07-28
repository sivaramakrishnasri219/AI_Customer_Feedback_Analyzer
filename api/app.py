"""
FastAPI Backend
"""

from fastapi import FastAPI
from pydantic import BaseModel

from preprocessing.clean_text import TextCleaner
from preprocessing.tokenizer import TextTokenizer
from preprocessing.lemmatizer import TextLemmatizer
from sentiment.sentiment_model import SentimentAnalyzer
from ner.ner_model import NamedEntityRecognizer
from chatbot.chatbot import CustomerFeedbackChatbot

##########################################################

app = FastAPI(
    title="AI Customer Feedback Analyzer",
    version="1.0"
)

##########################################################
# Load models only once
##########################################################

print("Loading Models...")

cleaner = TextCleaner()

tokenizer = TextTokenizer()

lemmatizer = TextLemmatizer()

sentiment = SentimentAnalyzer()

ner = NamedEntityRecognizer()

chatbot = CustomerFeedbackChatbot()

print("All Models Loaded Successfully!")

##########################################################
# Request Models
##########################################################

class ReviewRequest(BaseModel):

    review: str


class SearchRequest(BaseModel):

    question: str

##########################################################
# Home API
##########################################################

@app.get("/")

def home():

    return {

        "message": "AI Customer Feedback Analyzer API"

    }

##########################################################
# Analyze API
##########################################################

@app.post("/analyze")

def analyze(request: ReviewRequest):

    clean = cleaner.clean(request.review)

    tokens = tokenizer.nltk_tokenize(clean)

    tokens = tokenizer.remove_stopwords(tokens)

    lemmas = lemmatizer.nltk_lemmatize(tokens)

    sentiment_result = sentiment.predict(clean)

    entities = ner.extract_entities(request.review)

    return {

        "original_review": request.review,

        "clean_text": clean,

        "tokens": tokens,

        "lemmas": lemmas,

        "sentiment": sentiment_result,

        "entities": entities

    }

##########################################################
# Search API
##########################################################

@app.post("/search")

def search(request: SearchRequest):

    result = chatbot.search(

        request.question,

        top_k=5

    )

    return {

        "question": request.question,

        "results": result.to_dict(

            orient="records"

        )

    }

##########################################################
# Chat API
##########################################################

@app.post("/chat")

def chat(request: SearchRequest):

    result = chatbot.search(

        request.question,

        top_k=5

    )

    context = "\n".join(

        result["review"].tolist()

    )

    return {

        "question": request.question,

        "context": context,

        "message":

        "LLM integration will generate the final answer here."

    }