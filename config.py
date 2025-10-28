import os
from dotenv import load_dotenv

load_dotenv()

AZURE_API_KEY = os.getenv("AZURE_API_KEY")
AZURE_ENDPOINT = os.getenv("AZURE_ENDPOINT")
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX = os.getenv("PINECONE_INDEX", "rag-index")
