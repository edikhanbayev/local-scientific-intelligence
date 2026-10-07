import requests

from app.config import OLLAMA_URL, OLLAMA_MODEL

from app.vector_store import search_documents

def call_ollama(messages):
    response = requests.post(
        f"{OLLAMA_URL}/api/chat",
        json={
            "model": OLLAMA_MODEL,
            "messages": messages,
            "stream": False,
            "think": False,
            "options": {
                "temperature": 0.2,
                "num_predict": 200
            }
        },
        timeout=300
    )

    response.raise_for_status()

    data = response.json()

    if "message" not in data:
        raise RuntimeError(
            f"Ollama returned unexpected response: {data}"
        )

    return data["message"]["content"]

def ask_question(
    question: str,
    top_k: int = 4
):
    retrieved = search_documents(
        query=question,
        top_k=top_k
    )
    context_parts = []
    sources = []

    for item in retrieved:
        metadata = item["metadata"]
        source = metadata["source"]
        page = metadata["page"]
        context_parts.append(
            f"""
SOURCE: {source}
PAGE: {page}

{item["text"]}
"""
        )

        sources.append({
            "source": source,
            "page": page,
            "distance": item["distance"]
        })

    context = "\n\n".join(
        context_parts
    )

    system_prompt = """
Ты AI-ассистент для анализа научных
и технических документов.

Отвечай ТОЛЬКО на основе переданного
контекста.

Не используй внешние знания.

Если информации недостаточно,
прямо скажи:

"В предоставленных документах
недостаточно информации для ответа."

Каждое значимое утверждение
подкрепляй ссылкой в формате:

[filename, стр. X]
"""

    user_prompt = f"""
КОНТЕКСТ:

{context}

ВОПРОС:

{question}
"""

    answer = call_ollama([
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": user_prompt
        }
    ])

    return {
        "answer": answer,
        "sources": sources
    }