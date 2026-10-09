import os
import streamlit as st
import requests

st.title("Job Description RAG Assistant")

question = st.text_input("Ask your question")

if st.button("Submit"):
    if question:
        api_url = os.getenv("API_URL", "").rstrip("/")

        if not api_url:
            st.error("Backend API URL is not configured.")
        else:
            try:
                response = requests.post(f"{api_url}/ask",json={"question": question},timeout=120)
                response.raise_for_status()
                st.write(response.json()["answer"])

            except requests.RequestException as e:
                st.error(f"Request failed: {e}")

