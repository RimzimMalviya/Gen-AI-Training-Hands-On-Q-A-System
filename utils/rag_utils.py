import uuid
import PyPDF2
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone import Pinecone, ServerlessSpec
from config import Config

# Modern LangChain embedding imports
from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings


class PineconeRAGManager:
    def __init__(self, config: Config):
        self.config = config
        self.pc = None
        self.index = None
        self.embeddings = None
        self.active = False
        self._initialize()

    # ----------------------------------------------------------
    # Initialization logic
    # ----------------------------------------------------------
    def _initialize(self):
        if not self.config.PINECONE_API_KEY:
            print("Pinecone API key not found — using fallback keyword mode.")
            return

        try:
            self.pc = Pinecone(api_key=self.config.PINECONE_API_KEY)
            expected_dim = self._get_embedding_dimension()

            # Check if index exists and matches dimension
            existing_indexes = [i.name for i in self.pc.list_indexes()]
            if self.config.PINECONE_INDEX_NAME in existing_indexes:
                info = self.pc.describe_index(self.config.PINECONE_INDEX_NAME)
                if info.dimension != expected_dim:
                    print(f"Dimension mismatch! Pinecone index: {info.dimension}, expected: {expected_dim}")
                    print("Recreating index with correct dimension...")
                    self.pc.delete_index(self.config.PINECONE_INDEX_NAME)
                    self._create_index(expected_dim)
                else:
                    print(f"Pinecone index dimension OK ({expected_dim})")
            else:
                self._create_index(expected_dim)

            self.index = self.pc.Index(self.config.PINECONE_INDEX_NAME)

            # Initialize embeddings
            self.embeddings = self._init_embeddings()
            if not self.embeddings:
                print("Failed to initialize embeddings — RAG disabled.")
                return

            print("Pinecone RAG Manager ready")

        except Exception as e:
            print(f"Pinecone init failed: {e}")

    # ----------------------------------------------------------
    # Create Pinecone index
    # ----------------------------------------------------------
    def _create_index(self, dimension):
        try:
            print(f"Creating Pinecone index '{self.config.PINECONE_INDEX_NAME}' with dimension {dimension}")
            self.pc.create_index(
                name=self.config.PINECONE_INDEX_NAME,
                dimension=dimension,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )
        except Exception as e:
            print(f"Index creation failed: {e}")

    # ----------------------------------------------------------
    # Initialize embedding model
    # ----------------------------------------------------------
    def _init_embeddings(self):
        try:
            # Try Hugging Face first if available
            if self.config.HUGGINGFACEHUB_API_TOKEN:
                print(f"Using Hugging Face embeddings: {self.config.HUGGINGFACE_MODEL}")
                return HuggingFaceEmbeddings(model_name=self.config.HUGGINGFACE_MODEL)

            # Otherwise use Azure OpenAI embeddings
            print(f"Using Azure OpenAI embeddings: {self.config.EMBEDDING_MODEL}")
            return AzureOpenAIEmbeddings(
                model=self.config.EMBEDDING_MODEL,
                azure_endpoint=self.config.AZURE_OPENAI_ENDPOINT,
                api_key=self.config.AZURE_OPENAI_API_KEY,
                openai_api_version=self.config.AZURE_OPENAI_API_VERSION
            )
        except Exception as e:
            print(f"Embedding init failed: {e}")
            return None

    # ----------------------------------------------------------
    # Determine embedding dimension based on model name
    # ----------------------------------------------------------
    def _get_embedding_dimension(self):
        model = self.config.EMBEDDING_MODEL.lower()
        if "text-embedding-3-small" in model or "ada-002" in model:
            return 1536
        elif "3-large" in model:
            return 3072
        elif "mpnet" in model:
            return 768
        elif "minilm" in model:
            return 384
        else:
            return 1536  # Default safe fallback

    # ----------------------------------------------------------
    # Process & store document embeddings
    # ----------------------------------------------------------
    def process_document(self, filepath):
        """Extract PDF text, split into chunks, embed, and upsert into Pinecone."""
        try:
            pdf = PyPDF2.PdfReader(filepath)
            text = "\n".join(page.extract_text() for page in pdf.pages if page.extract_text())

            if not text.strip():
                print("PDF appears to have no readable text.")
                return False

            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            chunks = splitter.split_text(text)

            vectors = []
            for chunk in chunks:
                eid = str(uuid.uuid4())
                embedding = self.embeddings.embed_query(chunk)
                vectors.append((eid, embedding, {"text": chunk}))

            if self.index:
                self.index.upsert(vectors)
                print(f"📘 Uploaded {len(vectors)} embeddings to Pinecone.")
                self.active = True
                return True
            else:
                print("Pinecone not active, cannot store embeddings.")
                return False

        except Exception as e:
            print(f"Error processing document: {e}")
            return False

    # ----------------------------------------------------------
    # Query Pinecone for relevant document chunks
    # ----------------------------------------------------------
    def query_documents(self, query, k=3):
        if not self.active or not self.index:
            print("RAG not active or index unavailable.")
            return []

        try:
            q_emb = self.embeddings.embed_query(query)
            res = self.index.query(vector=q_emb, top_k=k, include_metadata=True)

            if hasattr(res, "matches"):
                return [m.metadata["text"] for m in res.matches if "text" in m.metadata]
            return []
        except Exception as e:
            print(f"Query error: {e}")
            return []

    # ----------------------------------------------------------
    # Helper flags
    # ----------------------------------------------------------
    def set_rag_enabled(self, enabled):
        self.active = enabled
        print(f"RAG {'enabled' if enabled else 'disabled'}")

    def is_rag_enabled(self):
        return self.active

    def get_document_info(self):
        return {
            "rag_enabled": self.active,
            "pinecone_index": self.config.PINECONE_INDEX_NAME
        }
