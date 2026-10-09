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

def retrieve(question):
    scores = cosine_similarity(
        vectorizer.transform([question]),
        tfidf_matrix
    ).flatten()

    ids = scores.argsort()[::-1][:4]

    relevant_ids = [
        i for i in ids if scores[i] > 0
    ]

    context = []
    for i in relevant_ids:
        chunk = chunks[i]
        metadata = chunk.metadata

        context.append(
            f"Job Title: {metadata.get('job_title', '')}\n"
            f"Category: {metadata.get('category', '')}\n"
            f"Skills: {metadata.get('job_skill_set', '')}\n"
            f"Description: {chunk.page_content}"
        )
    return "\n\n".join(context)

prompt = ChatPromptTemplate.from_template("""
Answer using only the context. If unavailable, say so.

Context: {context}
Question: {question}
""")

rag_chain = (
    {
        "context": RunnableLambda(retrieve),
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