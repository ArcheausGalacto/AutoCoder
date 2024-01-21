import sys
import os
from PyPDF2 import PdfReader
import openai

def read_pdf(file_path):
    """
    Reads a PDF file and extracts text from it.

    Parameters:
    file_path (str): The path to the PDF file.

    Returns:
    str: The extracted text from the PDF.
    """
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

def get_api_key(file_path):
    """
    Retrieves the OpenAI API key from a text file.

    Parameters:
    file_path (str): The path to the file containing the API key.

    Returns:
    str: The OpenAI API key.
    """
    with open(file_path, 'r') as file:
        api_key = file.read().strip()
    return api_key

def summarize_text(text, api_key):
    """
    Summarizes the given text using the specified GPT-4 API.

    Parameters:
    text (str): The text to summarize.
    api_key (str): The API key for authentication with OpenAI.

    Returns:
    str: The summary of the text using the specified GPT-4 API.
    """
    openai.api_key = api_key

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4-1106-preview",
            messages=[
                {"role": "system", "content": "You are a helpful assistant. Summarize the following text."},
                {"role": "user", "content": text}
            ],
        )

        summary = response['choices'][0]['message']['content']
        return summary
    except openai.error.OpenAIError as e:
        print(f"An error occurred: {e}")
        return None

def main(pdf_path, api_key_path):
    """
    Main function that orchestrates reading a PDF, extracting text,
    and summarizing it using OpenAI GPT-4.

    Parameters:
    pdf_path (str): The path to the PDF file to process.
    api_key_path (str): The path to the file containing the OpenAI API key.

    Returns:
    None
    """
    # Read and extract text from the PDF
    extracted_text = read_pdf(pdf_path)
    
    # Retrieve the API key from the file
    api_key = get_api_key(api_key_path)
    
    # Get summary of the extracted text using specified GPT-4 API
    summary = summarize_text(extracted_text, api_key)
    
    # Output the summary
    if summary:
        print(summary)
    else:
        print("Failed to generate summary.")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script.py <path_to_pdf> <path_to_api_key_file>")
        sys.exit(1)

    pdf_path = sys.argv[1]
    api_key_path = sys.argv[2]
    
    main(pdf_path, api_key_path)