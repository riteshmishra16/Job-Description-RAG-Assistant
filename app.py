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


# Load environment variables
load_dotenv()

# Load the saved TF-IDF index
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
index_path = os.path.join(BASE_DIR, "tfidf_index.joblib")

index = joblib.load(index_path)

vectorizer = index["vectorizer"]
tfidf_matrix = index["matrix"]
chunks = index["chunks"]


# Configure Gemini
api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    raise ValueError("GOOGLE_API_KEY is missing.")

model = ChatGoogleGenerativeAI(model="gemini-3.5-flash",google_api_key=api_key,temperature=0)


# Retrieve relevant job descriptions
def retrieve(question):

    question_vector = vectorizer.transform([question])
    scores = cosine_similarity(question_vector,tfidf_matrix).flatten()

    top_indices = scores.argsort()[::-1][:6]

    context = []

    for i in top_indices:

        if scores[i] <= 0:
            continue

        chunk = chunks[i]
        metadata = chunk.metadata

        context.append(
            f"Job Title: {metadata.get('job_title', '')}\n"
            f"Category: {metadata.get('category', '')}\n"
            f"Skills: {metadata.get('job_skill_set', '')}\n"
            f"Description: {chunk.page_content}"
        )

    result = "\n\n".join(context)

    # Debug information for Render logs
    print("Question:", question)
    print("Top similarity score:", float(scores[top_indices[0]]))
    print("Retrieved context length:", len(result))

    return result


# Answer generation prompt
prompt = ChatPromptTemplate.from_template("""
You are a helpful Job Description Assistant.

Answer the question using the relevant information in the context.

Rules:
- Give only the final answer in clear, simple language.
- Do not display metadata labels such as Job Title, Category,
  Skills, or Description.
- Do not repeat the full context.
- Use relevant skills and responsibilities from the job descriptions.
- Combine information from multiple relevant descriptions when useful.
- Do not invent facts or unsupported details.

Context:{context}

Question:{question}

Final answer:
""")


# RAG chain: retrieval -> prompt -> Gemini -> answer
rag_chain = (
    {
        "context": RunnableLambda(retrieve),
        "question": RunnablePassthrough()
    }| prompt| model| StrOutputParser()
)


# FastAPI application
app = FastAPI(title="Job Description RAG Assistant")


class Question(BaseModel):
    question: str


@app.get("/")
def home():
    return {"message": "RAG API is running"}

@app.post("/ask")
def ask(data: Question):
    answer = rag_chain.invoke(data.question)
    return {"answer": answer}
