
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os

load_dotenv()

app = FastAPI(title="Job Description RAG Assistant")

# Load FAISS database

os.environ["TOKENIZERS_PARALLELISM"] = "false"

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"batch_size": 1}
)

vector_store = FAISS.load_local(
    "faiss_index",
    embeddings,
    allow_dangerous_deserialization=True
)

retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 4}
)

prompt = ChatPromptTemplate.from_template(  """
    Answer the question using only the given context.
    Return only the direct answer.
    Do not include greetings, introductions, signatures,
    or closing statements.
    If the answer is unavailable in the context, say so. 
    Context: {context}
    Question: {question}""" )

model = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash"
)

# Build the RAG pipeline:
rag_chain = (
    {
        "context": retriever ,
        "question": RunnablePassthrough()
    }
    |prompt
    | model
)

print("RAG pipeline loaded successfully!")


class Question(BaseModel):
    question: str

# Home route
@app.get("/")
def home():
    return {"message": "Job Description RAG API is working"}


@app.post("/ask")
def ask_question(data: Question):
    try:
        response = rag_chain.invoke(data.question)

        return {
            "question": data.question,
            "answer": response.content[0]["text"]
        }

    except Exception as e:
       raise HTTPException(status_code=500, detail=str(e))