import streamlit as st
from transformers import pipeline
import PyPDF2
import os

# Load the summarizer pipeline
def load_summarizer():
    return pipeline("summarization", model="facebook/bart-large-cnn")

# Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in range(len(reader.pages)):
        text += reader.pages[page].extract_text()
    return text

# Function to summarize the extracted text from the PDF
def extract_and_summarize(pdf_text):
    summarizer = load_summarizer()

    # Set the maximum length dynamically based on the input text length
    input_length = len(pdf_text.split())
    max_length = min(500, input_length)  # Ensure max_length is reasonable

    # Generate the summary using the adjusted max_length
    summary = summarizer(pdf_text, max_length=max_length, min_length=50, do_sample=False)
    return summary[0]['summary_text']

# Streamlit UI
def main():
    st.title("PDF Text Summarizer")

    st.markdown("""
    Upload your research paper in PDF format, and this app will extract useful information
    and summarize the text for you.
    """)

    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    
    if uploaded_file is not None:
        # Display the PDF filename
        st.write("Filename:", uploaded_file.name)

        # Extract text from PDF
        with st.spinner("Extracting text from PDF..."):
            pdf_text = extract_text_from_pdf(uploaded_file)
        
        if len(pdf_text) > 0:
            # Display part of the extracted text
            st.subheader("Extracted Text (First 1000 characters):")
            st.text(pdf_text[:1000])  # Show first 1000 characters
            
            # Generate and display the summary
            st.subheader("Summary:")
            summary = extract_and_summarize(pdf_text)
            st.write(summary)
        else:
            st.error("No text extracted from the PDF. Please check the file content.")
    
if __name__ == "__main__":
    main()
