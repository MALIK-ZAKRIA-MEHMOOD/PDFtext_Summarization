import streamlit as st
from PyPDF2 import PdfReader
from transformers import pipeline

# Function to load the summarizer
@st.cache_resource
def load_summarizer():
    try:
        return pipeline("summarization", model="facebook/bart-large-cnn")
    except Exception as e:
        st.error(f"Error loading the summarization model: {e}")
        return None

summarizer = load_summarizer()

# Function to extract text from a PDF
def extract_text_from_pdf(pdf_file):
    pdf_reader = PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# Function to summarize sections of the PDF
def summarize_text_in_chunks(text, summarizer, max_chunk_length=1024):
    # Split the text into smaller chunks
    chunks = [text[i:i+max_chunk_length] for i in range(0, len(text), max_chunk_length)]
    detailed_summaries = []

    for i, chunk in enumerate(chunks):
        try:
            summary = summarizer(chunk, max_length=300, min_length=100, do_sample=False)[0]['summary_text']
            detailed_summaries.append(f"**Section {i+1} Summary:**\n{summary}\n")
        except Exception as e:
            detailed_summaries.append(f"Error summarizing section {i+1}: {e}")
    return "\n".join(detailed_summaries)

# Streamlit App
st.title("Research Paper Detailed Information Extractor")
st.write("Upload a research paper (PDF) to extract detailed and useful information.")

# File uploader
uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])

if uploaded_file is not None:
    st.write("Processing the uploaded PDF...")
    try:
        # Extract text from the PDF
        paper_text = extract_text_from_pdf(uploaded_file)
        
        if paper_text.strip():  # Ensure the extracted text is not empty
            st.success("PDF text extracted successfully!")
            st.write("Generating detailed summary, please wait...")

            # Generate detailed summaries
            if summarizer:
                detailed_summary = summarize_text_in_chunks(paper_text, summarizer)
                st.subheader("Detailed Extracted Summary:")
                st.write(detailed_summary)
            else:
                st.error("The summarizer model could not be loaded.")
        else:
            st.error("No text could be extracted from the PDF. Please ensure it contains text.")
    except Exception as e:
        st.error(f"An error occurred while processing the PDF: {e}")
