import os
import glob
import chromadb
from chromadb.utils import embedding_functions

PERSIST_DIR = "./chroma_db"
WORKPLACE_POLICIES_DIR = "./data/workplace_policies"
COLLECTION_NAME = "adtech_workplace_policies"


def build_workplace_vector_store():
    os.makedirs(PERSIST_DIR, exist_ok=True)
    client = chromadb.PersistentClient(path=PERSIST_DIR)

    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=ef
    )

    documents = []
    ids = []
    metadatas = []

    file_paths = glob.glob(os.path.join(WORKPLACE_POLICIES_DIR, "*.txt"))
    for path in file_paths:
        filename = os.path.basename(path)
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()

        chunks = [c.strip() for c in content.split("\n\n") if c.strip()]
        for chunk_idx, chunk in enumerate(chunks):
            documents.append(chunk)
            ids.append(f"{filename}_chunk_{chunk_idx}")
            metadatas.append({"source": filename, "domain": "adtech_workplace"})

    if documents:
        collection.add(documents=documents, ids=ids, metadatas=metadatas)
        print(f"Ingested {len(documents)} workplace policy chunks into collection '{COLLECTION_NAME}'.")


if __name__ == "__main__":
    build_workplace_vector_store()