import fitz  # PyMuPDF
from langchain_pinecone import PineconeVectorStore
from langchain_community.embeddings import HuggingFaceEmbeddings
from pinecone import Pinecone

def extract_text_from_pdf(pdf_path):
    text = ""
    with fitz.open(pdf_path) as doc:
        for page in doc:
            text += page.get_text()
    return text

def store_pdf_in_pinecone(pdf_path, index_name, api_key):
    pc = Pinecone(api_key=api_key)
    index = pc.Index(index_name)
    text = extract_text_from_pdf(pdf_path)
    embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = PineconeVectorStore(index=index, embedding=embedder)
    vectorstore.add_texts([text])
    return "PDF uploaded and indexed successfully."

def query_rag(question, index_name, api_key):
    pc = Pinecone(api_key=api_key)
    index = pc.Index(index_name)
    embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = PineconeVectorStore(index=index, embedding=embedder)
    docs = vectorstore.similarity_search(question, k=3)
    return "\n".join([d.page_content for d in docs])
