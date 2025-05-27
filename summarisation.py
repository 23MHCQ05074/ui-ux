import pdfplumber
from transformers import pipeline

def extract_text_from_pdf(pdf_path):
    text = ""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                extracted_text = page.extract_text()
                if extracted_text:
                    text += extracted_text + "\n"
    except Exception as e:
        print(f"Error reading PDF: {e}")
    return text.strip()

def summarize_text(text, max_length=500, min_length=100):
    try:
        summarizer = pipeline("summarization", model="facebook/bart-large-cnn", tokenizer="facebook/bart-large-cnn")
        if len(text) > 1024:
            text = text[:1024]  # Truncate text to avoid exceeding token limits
        summarized_text = summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)[0]['summary_text']
        return summarized_text
    except Exception as e:
        print(f"Error during summarization: {e}")
        return ""

def main(pdf_file_path):
    text = extract_text_from_pdf(pdf_file_path)
    if not text:
        print("No text found in the PDF.")
        return

    summary = summarize_text(text)
    print("\n--- Summary ---")
    print(summary)

if __name__ == "__main__":
    pdf_file = "path/to/your/pdf_file.pdf"
    main(pdf_file)