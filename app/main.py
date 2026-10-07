from fastapi import FastAPI

from app.schemas import QuestionRequest, TextRequest, ClusterRequest
from app.rag import ask_question
from app.nlp import extract_entities
from app.topics import classify_topic, cluster_texts
from app.relations import extract_relations

app = FastAPI(
    title=(
        "Local Scientific "
        "Intelligence Assistant"
    ),
    description=(
        "Local RAG and NLP system "
        "without external LLM APIs"
    ),
    version="1.0.0"
)

@app.get("/health")
def health():
    return {
        "status": "ok"
    }

@app.post("/ask")
def ask(request: QuestionRequest):
    return ask_question(
        question=request.question,
        top_k=request.top_k
    )

@app.post("/ner")
def ner(request: TextRequest):
    return {
        "entities":
            extract_entities(
                request.text
            )
    }

@app.post("/classify")
def classify(request: TextRequest):
    return classify_topic(
        request.text
    )

@app.post("/relations")
def relations(request: TextRequest):
    return {
        "relations":
            extract_relations(
                request.text
            )
    }

@app.post("/cluster")
def cluster(
    request: ClusterRequest
):
    return {
        "clusters":
            cluster_texts(
                texts=request.texts,
                n_clusters=(
                    request.n_clusters
                )
            )
    }