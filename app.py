import os
import joblib
from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

index = joblib.load("tfidf_index.joblib")
vectorizer = index["vectorizer"]
tfidf_matrix = index["matrix"]
chunks = index["chunks"]

model = ChatGoogleGenerativeAI(model="gemini-3.5-flash", google_api_key=os.getenv("GOOGLE_API_KEY"), temperature=0)

def retrieve_context(question):
    scores = cosine_similarity(vectorizer.transform([question]), tfidf_matrix).flatten()
    top_indices = scores.argsort()[::-1][:4]

    return "\n\n".join(chunks[i].page_content
        for i in top_indices if scores[i] > 0
    )

prompt = ChatPromptTemplate.from_template("""
Answer using only the context. If unavailable, say so.

Context: {context}
Question: {question}
""")

rag_chain = (
    {
        "context": RunnableLambda(retrieve_context),
        "question": RunnablePassthrough()
    }| prompt| model| StrOutputParser()
)

app = FastAPI()

class Question(BaseModel):
    question: str

@app.get("/")
def home():
    return {"message": "RAG API is running"}

@app.post("/ask")
def ask(data: Question):
    return {"answer": rag_chain.invoke(data.question)}