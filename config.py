import os
import secrets
from dotenv import load_dotenv

load_dotenv()

class Config:
    # Flask Configuration
    SECRET_KEY = os.getenv('SECRET_KEY', secrets.token_hex(16))
    
    # File Upload Configuration
    UPLOAD_FOLDER = 'uploads'
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size
    
    # Azure OpenAI Configuration
    AZURE_OPENAI_API_KEY = os.getenv('AZURE_OPENAI_API_KEY')
    AZURE_OPENAI_ENDPOINT = os.getenv('AZURE_OPENAI_ENDPOINT')
    AZURE_OPENAI_API_VERSION = os.getenv('AZURE_OPENAI_API_VERSION', '2023-12-01-preview')
    AZURE_DEPLOYMENT_NAME = os.getenv('AZURE_DEPLOYMENT_NAME', 'gpt-35-turbo')
    AZURE_EMBEDDING_DEPLOYMENT_NAME = os.getenv('AZURE_EMBEDDING_DEPLOYMENT_NAME', 'text-embedding-ada-002')
    
    # Hugging Face Configuration
    HUGGINGFACEHUB_API_TOKEN = os.getenv('HUGGINGFACEHUB_API_TOKEN')
    HUGGINGFACE_MODEL = os.getenv('HUGGINGFACE_MODEL', 'google/flan-t5-base')
    
    # Pinecone Configuration
    PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
    PINECONE_ENVIRONMENT = os.getenv('PINECONE_ENVIRONMENT')
    PINECONE_INDEX_NAME = os.getenv('PINECONE_INDEX_NAME', 'chatbot-documents')
    
    # Model Settings
    MAX_TOKENS = int(os.getenv('MAX_TOKENS', 1000))
    TEMPERATURE = float(os.getenv('TEMPERATURE', 0.7))
    
    @classmethod
    def print_config_status(cls):
        """Print configuration status for debugging"""
        print("\n=== Configuration Status ===")
        print(f"Azure OpenAI API Key: {'Set' if cls.AZURE_OPENAI_API_KEY else 'Missing'}")
        print(f"Azure OpenAI Endpoint: {'Set' if cls.AZURE_OPENAI_ENDPOINT else 'Missing'}")
        print(f"Azure Deployment: {cls.AZURE_DEPLOYMENT_NAME}")
        print(f"Azure Embedding Deployment: {cls.AZURE_EMBEDDING_DEPLOYMENT_NAME}")
        print(f"Hugging Face Token: {'Set' if cls.HUGGINGFACEHUB_API_TOKEN else 'Missing'}")
        print(f"Pinecone: {'Set' if cls.PINECONE_API_KEY else 'Missing'}")
        
        if not cls.AZURE_OPENAI_API_KEY and not cls.HUGGINGFACEHUB_API_TOKEN:
            print("\n ERROR: No AI providers configured!")
        else:
            print("\n Configuration looks good!")
        print("============================\n")

# Print config status when module is imported
Config.print_config_status()