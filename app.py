import streamlit as st
def load_summarizer():
    return pipeline("summarization", model="facebook/bart-large-cnn")

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
    paper_text = extract_text_from_pdf(uploaded_file)
    
    if paper_text:
        st.write("Text extracted successfully! Generating summary...")
        # Summarize text
        summary = summarizer(paper_text[:1024], max_length=200, min_length=50, do_sample=False)[0]['summary_text']
        st.subheader("Summary:")
        st.write(summary)
    else:
        st.error("Could not extract text from the uploaded PDF. Please ensure it's a valid PDF.")

