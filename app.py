import streamlit as st
from PyPDF2 import PdfReader
from transformers import pipeline

# Function to load the summarization pipeline
@st.cache_resource
def load_summarizer():
    try:
        summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
        return summarizer
    except Exception as e:
        st.error(f"Error loading the model: {e}")
        return None

summarizer = load_summarizer()

# Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    pdf_reader = PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# App interface
st.title("Research Paper Information Extractor")
st.write("Upload a research paper (PDF) to extract and summarize useful information.")

uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])

if uploaded_file is not None:
    st.write("Extracting text from the PDF...")
    try:
        paper_text = extract_text_from_pdf(uploaded_file)
        if paper_text.strip():  # Check if extracted text is not empty
            st.write("Text extracted successfully! Generating summary...")
            if summarizer:
                # Summarize text
                summary = summarizer(
                    paper_text[:1024],  # Limit input text to the model's capacity
                    max_length=200,
                    min_length=50,
                    do_sample=False
                )[0]['summary_text']
                st.subheader("Summary:")
                st.write(summary)
            else:
                st.error("Summarization model is not available.")
        else:
            st.error("No text was extracted from the uploaded PDF. Please ensure it's a valid PDF.")
    except Exception as e:
        st.error(f"An error occurred: {e}")
