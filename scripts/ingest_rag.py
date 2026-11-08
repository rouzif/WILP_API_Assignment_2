import os
import glob
import chromadb
from chromadb.utils import embedding_functions

PERSIST_DIR = "./chroma_db"
POLICIES_DIR = "./data/policies"
COLLECTION_NAME = "shopease_policies"


def build_vector_store():
    os.makedirs(PERSIST_DIR, exist_ok=True)

    # Initialize Chroma persistent client
    client = chromadb.PersistentClient(path=PERSIST_DIR)

    # Use standard huggingface sentence-transformers model (runs locally & free)
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )

    # Get or create collection
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=ef
    )

    documents = []
    ids = []
    metadatas = []

    file_paths = glob.glob(os.path.join(POLICIES_DIR, "*.txt"))

    for idx, path in enumerate(file_paths):
        filename = os.path.basename(path)
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()

        # Basic paragraph chunking
        chunks = [c.strip() for c in content.split("\n\n") if c.strip()]
        for chunk_idx, chunk in enumerate(chunks):
            documents.append(chunk)
            ids.append(f"{filename}_chunk_{chunk_idx}")
            metadatas.append({"source": filename})

    if documents:
        collection.add(documents=documents, ids=ids, metadatas=metadatas)
        print(f"Successfully ingested {len(documents)} document chunks into ChromaDB at '{PERSIST_DIR}'.")


if __name__ == "__main__":
    build_vector_store()