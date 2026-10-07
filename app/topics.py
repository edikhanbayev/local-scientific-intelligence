from functools import lru_cache
import numpy as np
from sklearn.cluster import KMeans
from app.embeddings import encode_texts

TOPICS = {
    "artificial_intelligence":
        "искусственный интеллект машинное обучение "
        "нейронные сети computer vision NLP",

    "robotics":
        "робототехника автономные роботы "
        "industrial robotics automation",

    "energy":
        "энергетика батареи hydrogen solar "
        "energy storage electricity",

    "materials":
        "новые материалы металлургия composites "
        "nanomaterials",

    "biotechnology":
        "биотехнологии medicine genetics "
        "pharmaceutical",

    "telecommunications":
        "телекоммуникации 5G networks "
        "wireless communication",

    "cybersecurity":
        "кибербезопасность information security "
        "network attacks"
}

@lru_cache(maxsize=1)
def get_topic_embeddings():

    texts = list(
        TOPICS.values()
    )
    return np.array(
        encode_texts(texts)
    )

def classify_topic(text: str):

    text_embedding = np.array(
        encode_texts([text])[0]
    )
    topic_embeddings = (
        get_topic_embeddings()
    )

    # embeddings normalized,
    # dot product ≈ cosine similarity

    scores = topic_embeddings @ text_embedding

    best_index = int(
        np.argmax(scores)
    )

    labels = list(
        TOPICS.keys()
    )

    return {
        "label": labels[best_index],
        "score": float(
            scores[best_index]
        )
    }

def cluster_texts(
    texts: list[str],
    n_clusters: int = 4
):
    embeddings = np.array(
        encode_texts(texts)
    )
    model = KMeans(
        n_clusters=n_clusters,
        random_state=42,
        n_init="auto"
    )

    clusters = model.fit_predict(
        embeddings
    )

    return [
        {
            "text": text,
            "cluster": int(cluster)
        }
        for text, cluster
        in zip(texts, clusters)
    ]