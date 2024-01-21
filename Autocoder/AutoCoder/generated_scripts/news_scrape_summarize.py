import sys
import requests
from bs4 import BeautifulSoup
import openai

# Function to load the API key from a given file path.
def load_api_key(file_path):
    with open(file_path, 'r') as file:
        api_key = file.read().strip()
    return api_key

# Function to scrape the webpage and return its content.
def scrape_website(url):
    response = requests.get(url)
    if response.ok:
        return response.text
    else:
        return None

# Function to extract the relevant text from a webpage content.
def extract_text(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    # Assumption: The text is within <p> tags.
    paragraphs = soup.find_all('p')
    return ' '.join(paragraph.text for paragraph in paragraphs)

# Function to use OpenAI's ChatGPT to summarize the text content.
def summarize_text_with_gpt(api_key, text):
    openai.api_key = api_key
    
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "system", "content": "You are a helpful assistant. Summarize the following article."},
                  {"role": "user", "content": text}],
    )
    
    return response['choices'][0]['message']['content']

def main(url):
    api_key_path = "F:\\AutoCoder\\api_key.txt"
    api_key = load_api_key(api_key_path)

    html_content = scrape_website(url)
    if html_content:
        extracted_text = extract_text(html_content)
        summary = summarize_text_with_gpt(api_key, extracted_text)
        if summary:
            print(summary)
        else:
            print("Failed to summarize the text.")
    else:
        print("Failed to retrieve the webpage content.")

if __name__ == "__main__":
    # Check if a command line argument (URL) was provided.
    if len(sys.argv) != 2:
        print("Usage: python script.py <url>")
        sys.exit(1)
    url = sys.argv[1]
    main(url)