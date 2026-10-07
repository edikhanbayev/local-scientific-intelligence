import chromadb

from app.config import CHROMA_DIR,COLLECTION_NAME
from app.embeddings import encode_texts

client = chromadb.PersistentClient(
    path=str(CHROMA_DIR)
)

collection = client.get_or_create_collection(
    name=COLLECTION_NAME
)

def add_documents(records):
    texts = [
        record["text"]
        for record in records
    ]
    embeddings = encode_texts(texts)
    collection.upsert(
        ids=[
            record["id"]
            for record in records
        ],
        documents=texts,
        embeddings=embeddings,
        metadatas=[
            record["metadata"]
            for record in records
        ]
    )

def search_documents(
    query: str,
    top_k: int = 4
):

    query_embedding = encode_texts(
        [query]
    )[0]

    result = collection.query(
        query_embeddings=[
            query_embedding
        ],
        n_results=top_k,
        include=[
            "documents",
            "metadatas",
            "distances"
        ]
    )

    items = []

    documents = result["documents"][0]
    metadatas = result["metadatas"][0]
    distances = result["distances"][0]

    for document, metadata, distance in zip(
        documents,
        metadatas,
        distances
    ):
        items.append({
            "text": document,
            "metadata": metadata,
            "distance": distance
        })
    return items