# Job Description RAG Assistant

A Retrieval-Augmented Generation (RAG) application that answers questions about job descriptions and required skills using a job-related dataset.

## Features

* Retrieves relevant job information using TF-IDF and cosine similarity.
* Uses LangChain Documents and RecursiveCharacterTextSplitter.
* Generates context-based answers using Google Gemini.
* Uses LangChain LCEL to build the RAG pipeline.
* Provides a FastAPI backend.
* Provides a Streamlit user interface.

## Tech Stack

* Python
* LangChain
* Scikit-learn (TF-IDF and cosine similarity)
* Google Gemini
* FastAPI
* Streamlit
* Hugging Face Datasets

## Dataset

The project uses the [Job Skill Set dataset](https://huggingface.co/datasets/batuhanmtl/job-skill-set) from Hugging Face.

## Project Structure

```text
JobDescription Project/
├── app.py
├── streamlit.py
├── rag_pipeline.ipynb
├── tfidf_index.joblib
├── requirements.txt
├── .gitignore
└── README.md
```

## How It Works

1. Job descriptions are converted into LangChain Document objects.
2. RecursiveCharacterTextSplitter divides documents into smaller text chunks.
3. TF-IDF converts the chunks into numerical vectors.
4. Cosine similarity retrieves relevant chunks for a question.
5. LangChain passes the retrieved context and question to the Gemini model.
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

## Run the Application

Start the FastAPI backend:

```bash
python -m uvicorn app:app --reload
```

Open another terminal and start the Streamlit interface:

```bash
python -m streamlit run streamlit_app.py
```

API documentation:

http://127.0.0.1:8000/docs

## Example Questions

* What skills are required for data science jobs?
* What are the responsibilities of a Data Scientist?
* What skills are required for finance jobs?

## Author

**Ritesh Kumar Mishra**

GitHub: https://github.com/riteshmishra16
