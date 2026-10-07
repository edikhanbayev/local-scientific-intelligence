from pydantic import BaseModel, Field

class QuestionRequest(BaseModel):
    question: str
    top_k: int = Field(
        default=4,
        ge=1,
        le=10
    )

class TextRequest(BaseModel):
    text: str

class ClusterRequest(BaseModel):
    texts: list[str]
    n_clusters: int = Field(
        default=3,
        ge=2,
        le=10
    )