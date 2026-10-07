from functools import lru_cache
from sentence_transformers import SentenceTransformer
from app.config import EMBEDDING_MODEL

@lru_cache(maxsize=1)
def get_embedding_model():
    print(
        f"Loading embedding model: "
        f"{EMBEDDING_MODEL}"
    )
    return SentenceTransformer(
        EMBEDDING_MODEL
    )

def encode_texts(texts: list[str]):
    model = get_embedding_model()
    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=False
    )
    return embeddings.tolist()