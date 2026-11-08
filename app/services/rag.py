import os
import chromadb
from chromadb.utils import embedding_functions

class RAGService:
    def __init__(self):
        self.persist_dir = "./chroma_db"
        self.collection_name = "shopease_policies"
        self.client = None
        self.collection = None

    def initialize(self):
        if self.collection is None and os.path.exists(self.persist_dir):
            self.client = chromadb.PersistentClient(path=self.persist_dir)
            ef = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name="all-MiniLM-L6-v2"
            )
            self.collection = self.client.get_collection(
                name=self.collection_name,
                embedding_function=ef
            )

    def retrieve_context(self, query: str, top_k: int = 2) -> str:
        self.initialize()
        if not self.collection:
            return ""

        results = self.collection.query(
            query_texts=[query],
            n_results=top_k
        )

        documents = results.get("documents", [[]])[0]
        if not documents:
            return ""

        context_str = "\n---\n".join(documents)
        return context_str

rag_service = RAGService()