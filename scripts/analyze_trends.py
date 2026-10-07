from collections import defaultdict

from app.vector_store import collection
from app.topics import classify_topic


def main():
    result = collection.get(
        include=[
            "documents",
            "metadatas"
        ]
    )
    # Собираем chunks по документам
    documents = defaultdict(
        lambda: {
            "texts": [],
            "metadata": None
        }
    )
    for text, metadata in zip(
        result["documents"],
        result["metadatas"]
    ):

        source = metadata["source"]
        documents[source][
            "texts"
        ].append(text)

        documents[source][
            "metadata"
        ] = metadata

    statistics = defaultdict(
        lambda: defaultdict(int)
    )

    for source, data in (
        documents.items()
    ):
        text = " ".join(
            data["texts"][:3]
        )
        topic = classify_topic(
            text
        )["label"]

        date = data[
            "metadata"
        ].get(
            "date",
            ""
        )

        if not date:
            continue

        year = date[:4]

        statistics[
            year
        ][topic] += 1

    print("\nTOPIC COUNTS\n")

    for year in sorted(
        statistics.keys()
    ):

        print(year)

        for topic, count in (
            statistics[year].items()
        ):

            print(
                f"  {topic}: {count}"
            )

if __name__ == "__main__":
    main()