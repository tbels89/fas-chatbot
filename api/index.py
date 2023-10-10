#live backend for vercel
import os
from dotenv import load_dotenv
from flask import Flask, request, jsonify
from langchain.document_loaders import DirectoryLoader, UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
import re
import urllib.request
import xml.etree.ElementTree as ET
import requests
from typing import List
from tenacity import retry, wait_exponential

app = Flask(__name__)
load_dotenv()

# API key for OpenAI
api_key = os.environ['OPENAI_API_KEY']  # Replace with your OpenAI API key

# Define model parameters
llm = ChatOpenAI(
    model_name="gpt-4",
    temperature=0.1,
    max_tokens=1900,
    n=1,
    openai_api_key=api_key
)

# Sitemap URL
sitemap_url = "https://famaservices.net/sitemap.xml"  # Replace with your sitemap URL

# Function to extract URLs from the sitemap
def extract_urls_from_sitemap(sitemap: str) -> List[str]:
    urls = []

    def process_sitemap(sitemap_url):
        response = requests.get(sitemap_url)
        sitemap_content = response.text

        # Extract URLs enclosed within <loc> tags
        extracted_urls = re.findall(r"<loc>(.*?)</loc>", sitemap_content)

        if not extracted_urls:
            # Extract URLs enclosed within <url> tags
            extracted_urls = re.findall(r"<url>\s*<loc>(.*?)</loc>\s*</url>", sitemap_content)

        if not extracted_urls:
            # Extract URLs separated by tabs
            extracted_urls = re.findall(r"\t(https?://[^\s]*)", sitemap_content)

        if not extracted_urls:
            # Extract URLs separated by line breaks
            extracted_urls = re.findall(r"\n(https?://[^\s]*)", sitemap_content)

        urls.extend(extracted_urls)

        # Check if the extracted URLs are sitemap URLs ending with .xml
        nested_sitemap_urls = [url for url in extracted_urls if url.endswith('.xml')]

        # Recursively process the nested sitemaps
        for nested_sitemap_url in nested_sitemap_urls:
            process_sitemap(nested_sitemap_url)

    # Attempt regular expression-based extraction first
    process_sitemap(sitemap)

    # If the URLs list is still empty, try using xml.etree.ElementTree as a fallback
    if not urls:
        try:
            response = urllib.request.urlopen(sitemap)
            sitemap_data = response.read()

            root = ET.fromstring(sitemap_data)
            for element in root.iter():
                if "url" in element.tag:
                    for loc in element:
                        if "loc" in loc.tag:
                            urls.append(loc.text)

                # Check if there are nested sitemaps
                if "sitemap" in element.tag:
                    for loc in element:
                        if "loc" in loc.tag:
                            try:
                                urls.extend(extract_urls_from_sitemap(loc.text))
                            except ET.ParseError:
                                # Skip this nested sitemap if it is not well-formed
                                continue

        except ET.ParseError:
            # Skip the current sitemap if it is not well-formed
            return []
    return urls

# Extract URLs from the sitemap
sitemap_urls = extract_urls_from_sitemap(sitemap_url)

# Folder path for additional data files (.txt, .pdf, .csv, .docx, .xlsx)
data_folder = "./data"  # Replace with your data folder path

# Load data from the sitemap and data folder
data_loader = DirectoryLoader(data_folder, glob='*.{txt,pdf,csv,docx,xlsx}')
url_loader = UnstructuredURLLoader(sitemap_urls)
data = [] 
data += data_loader.load()
data += url_loader.load()

# Create vector database of the loaded data
text_splitter = RecursiveCharacterTextSplitter(chunk_size=8000, chunk_overlap=3000)
split_data = text_splitter.split_documents(data)
embeddings = OpenAIEmbeddings(openai_api_key=api_key)
vector_data = Chroma.from_documents(split_data, embeddings)

# Define connection to model
qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=vector_data.as_retriever())

@retry(wait=wait_exponential(multiplier=0.02, max=32))
@app.route('/index', methods=['POST'])
def chatbot():
    input_text = request.json['input']
    history = request.json['history'] or []

    chat_list = list(sum(history, ()))
    chat_list.append(input_text)
    query = ' '.join(chat_list)

    response = qa({"query": input_text})
    response_text = response['result']

    # Check if a suitable response was found in the knowledge base
    if response_text.strip() == "":
        # Generate an independent response using the chatbot
        independent_response = qa({"query": "Provide an independent response to: " + input_text})
        response_text = independent_response['result']

        # Get the source information from the vector data for the independent response
        source = vector_data.get_source_for_response(response_text)
    else:
        # Get the source information from the vector data for the response
        source = vector_data.get_source_for_response(response_text)

    history.append((input_text, response_text, source))
    return jsonify({"history": history, "response": response_text, "source": source})


if __name__ == '__main__':
    app.run()