import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import Config

class SimpleRAGManager:
    def __init__(self, config):
        self.config = config
        self.document_chunks = []
        self.rag_enabled = False
        print("Simple RAG Manager initialized (text-based only)")
    
    def process_document(self, filepath):
        """Process PDF document for text-based search"""
        try:
            print(f"Processing document: {filepath}")
            
            # Load PDF document
            loader = PyPDFLoader(filepath)
            documents = loader.load()
            print(f"Loaded {len(documents)} pages")
            
            # Split documents into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            self.document_chunks = text_splitter.split_documents(documents)
            print(f"Split into {len(self.document_chunks)} chunks")
            print("Document processed successfully (text-based search)")
            
            return True
        
        except Exception as e:
            print(f"Error processing document: {e}")
            return False
    
    def query_documents(self, query, k=3):
        """Simple text-based document search"""
        if not self.document_chunks:
            print("No documents available for search")
            return []
        
        print(f"🔍 Searching {len(self.document_chunks)} chunks for: '{query}'")
        
        # Simple keyword matching with scoring
        query_words = [word.lower() for word in query.split() if len(word) > 2]  # Filter short words
        scored_results = []
        
        for i, chunk in enumerate(self.document_chunks):
            content = chunk.page_content.lower()
            
            # Calculate relevance score
            score = 0
            matching_words = []
            
            for word in query_words:
                if word in content:
                    score += 1
                    matching_words.append(word)
            
            if score > 0:
                scored_results.append({
                    'score': score,
                    'content': chunk.page_content,
                    'chunk_index': i,
                    'matching_words': matching_words
                })
                print(f"Chunk {i}: score {score}, matches: {matching_words}")
        
        # Sort by score and return top k
        scored_results.sort(key=lambda x: x['score'], reverse=True)
        results = [result['content'] for result in scored_results[:k]]
        
        print(f"Found {len(results)} relevant chunks out of {len(scored_results)} matches")
        
        # Show what we found
        for i, result in enumerate(results):
            preview = result[:100] + "..." if len(result) > 100 else result
            print(f"   {i+1}. {preview}")
        
        return results
    
    def set_rag_enabled(self, enabled):
        """Enable or disable RAG"""
        self.rag_enabled = enabled
        status = "enabled" if enabled else "disabled"
        print(f"RAG {status}")
    
    def is_rag_enabled(self):
        """Check if RAG is enabled"""
        return self.rag_enabled and len(self.document_chunks) > 0
    
    def get_document_info(self):
        """Get information about loaded documents"""
        return {
            'chunks_loaded': len(self.document_chunks),
            'rag_enabled': self.rag_enabled,
            'rag_ready': self.is_rag_enabled()
        }