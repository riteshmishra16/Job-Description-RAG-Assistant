# Job Description RAG Assistant

A Retrieval-Augmented Generation (RAG) application that answers questions about job descriptions and required skills using a job-related dataset.

## Features

* Retrieves relevant job information using semantic similarity search.
* Uses FAISS as a vector database.
* Generates context-based answers using Google Gemini.
* Uses LangChain to build the RAG pipeline.
* Provides a FastAPI backend.
* Provides a Streamlit user interface.

## Tech Stack

* Python
* LangChain
* FAISS
* Hugging Face Sentence Transformers
* Google Gemini
* FastAPI
* Streamlit

## Dataset

The project uses the [Job Skill Set dataset](https://huggingface.co/datasets/batuhanmtl/job-skill-set) from Hugging Face.

## Project Structure

```text
JobDescription Project/
├── app.py
├── streamlit.py
├── rag_pipeline.ipynb
├── faiss_index/
├── requirements.txt
├── .gitignore
└── README.md
```

## How It Works

1. Job descriptions are split into smaller text chunks.
2. Hugging Face embeddings convert text into numerical vectors.
3. FAISS stores the vectors for similarity search.
4. The retriever finds relevant documents for a question.
5. LangChain passes the retrieved context to the Gemini model.
6. Gemini generates an answer based on the retrieved context.
7. FastAPI serves the backend, and Streamlit provides the interface.

## Installation

Install the required packages:

```bash
pip install -r requirements.txt
```

Create a `.env` file and add your Google API key:

```text
GOOGLE_API_KEY=your_google_api_key
```

Do not upload your API key to GitHub.

## Run the Application

Start the FastAPI backend:

```bash
python -m uvicorn app:app --reload
```

Open another terminal and start the Streamlit interface:

```bash
python -m streamlit run streamlit.py
```

API documentation:

`http://127.0.0.1:8000/docs`

## Example Questions

* What skills are required for data science jobs?
* What are the responsibilities of a Data Scientist?
* What skills are required for finance jobs?

## Author

**Ritesh Kumar Mishra**

GitHub: https://github.com/riteshmishra16
