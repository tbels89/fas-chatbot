
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
    model_name="gpt-4", #gpt-4 only 10k tpm, not high enough limit to use this regularly 
    temperature=0.1,
    max_tokens=4000,
    n=1,
    openai_api_key=api_key
)

# Define model parameters for general knowledge
general_knowledge_model = ChatOpenAI(
    model_name="gpt-4",
    temperature=0.7,
    max_tokens=3000,
    n=1,
    openai_api_key=api_key
)


# Sitemap URL
sitemap_url = "https://famaservices.net/sitemap.xml" 

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
data_loader = DirectoryLoader(path=data_folder)
url_loader = UnstructuredURLLoader(sitemap_urls)
data = [] 
print("initializing...\n")

data += data_loader.load()
print("loading data...")
if len(data) > 0:
    print('directory data loaded successfully')
    print(len(data))
else:
    print('data not loaded successfully. whoops')

data += url_loader.load()
print('url data loaded')
print('all data loaded. creating database...\n')

# Create vector database of the loaded data
text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=1000) #changed chunk size but still seeing error 
print('text splitter done')
split_data = text_splitter.split_documents(data)
print('split data done')
embeddings = OpenAIEmbeddings(openai_api_key=api_key, model="text-embedding-ada-002") #will change embeddings model after to something free if possible 
print('embeddings done')
vector_data = Chroma.from_documents(split_data, embeddings)
print('vector data loaded')

# Define connection to model
print('define connection to model with qa')
qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=vector_data.as_retriever())
print('database created')

# Function to interact with the chatbot from the terminal
@retry(wait=wait_exponential(multiplier=0.02, max=32))
def interact_with_chatbot():
    while True:
        input_text = input("Enter your question (or 'q' to quit): ")
        if input_text.lower() == 'q':
            break

        response = qa({"query": input_text})
        response_text = f"{response['result']}"

        if response_text.strip() == "":
            # If no answer found in the vector database, consult general knowledge
            general_response = general_knowledge_model({"query": input_text})
            general_response_text = f"{general_response['result']}"

            if general_response_text.strip() != "":
                response_text = general_response_text
            else:
                response_text = "I couldn't find a suitable response in my knowledge base."

        print("Chatbot's Response:", response_text)

if __name__ == '__main__':
    interact_with_chatbot()
