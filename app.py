from transformers import pipeline

# Load the summarizer pipeline
def load_summarizer():
    return pipeline("summarization", model="facebook/bart-large-cnn")

# Function to extract and summarize text from the PDF
def extract_and_summarize(pdf_text):
    summarizer = load_summarizer()

    # Set the maximum length dynamically based on the input text length
    input_length = len(pdf_text.split())
    max_length = min(500, input_length)  # Ensure max_length is reasonable

    # Generate the summary using the adjusted max_length
    summary = summarizer(pdf_text, max_length=max_length, min_length=50, do_sample=False)
    return summary

# Example usage (assuming pdf_text is the text content extracted from the PDF)
pdf_text = "Your extracted PDF content here..."
summary = extract_and_summarize(pdf_text)
print(summary)
