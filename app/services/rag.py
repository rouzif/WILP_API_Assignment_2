import os
import chromadb
from chromadb.utils import embedding_functions


class RAGService:
    def __init__(self):
        self.persist_dir = "./chroma_db"
        self.client = None
        self.collections = {}

    def initialize(self):
        if self.client is None and os.path.exists(self.persist_dir):
            self.client = chromadb.PersistentClient(path=self.persist_dir)
            self.ef = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name="all-MiniLM-L6-v2"
            )
            for col_name in ["shopease_policies", "adtech_workplace_policies"]:
                try:
                    self.collections[col_name] = self.client.get_collection(
                        name=col_name, embedding_function=self.ef
                    )
                except Exception:
                    pass

    def retrieve_context(self, query: str, domain: str = "vanilla", top_k: int = 2) -> str:
        self.initialize()
        col_name = "adtech_workplace_policies" if domain == "workplace" else "shopease_policies"
        collection = self.collections.get(col_name)

        if not collection:
            return ""

        results = collection.query(query_texts=[query], n_results=top_k)
        documents = results.get("documents", [[]])[0]
        return "\n---\n".join(documents) if documents else ""


rag_service = RAGService()