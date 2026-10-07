import json
from pathlib import Path

from app.vector_store import search_documents

FILE = (
    Path(__file__)
    .parent
    / "questions.json"
)

with open(
    FILE,
    "r",
    encoding="utf-8"
) as f:
    cases = json.load(f)

hits = 0
reciprocal_ranks = []

for case in cases:
    results = search_documents(
        case["question"],
        top_k=2
    )
    rank = None
    for index, result in enumerate(
        results,
        start=1
    ):
        source = result[
            "metadata"
        ]["source"]
        if (
            source
            == case["expected_source"]
        ):
            rank = index
            break

    if rank is not None:
        hits += 1
        reciprocal_ranks.append(
            1 / rank
        )
    else:
        reciprocal_ranks.append(0)

hit_rate = (hits / len(cases))
mrr = (
    sum(reciprocal_ranks)
    /
    len(reciprocal_ranks)
)
print(
    f"HitRate@4: "
    f"{hit_rate:.3f}"
)

print(
    f"MRR@4: "
    f"{mrr:.3f}"
)