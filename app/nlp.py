from functools import lru_cache
from transformers import pipeline

@lru_cache(maxsize=1)
def get_ner_pipeline():
    return pipeline(
        task="token-classification",
        model=(
            "Babelscape/"
            "wikineural-multilingual-ner"
        ),
        aggregation_strategy="simple",
        device=-1
    )

def extract_entities(text: str):
    ner = get_ner_pipeline()
    results = ner(text)
    entities = []
    for item in results:

        entities.append({
            "entity": item["word"],
            "type": item["entity_group"],
            "score": float(
                item["score"]
            )
        })
    return entities