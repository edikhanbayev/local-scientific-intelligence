import json
from app.rag import call_ollama

def extract_relations(text: str):

    prompt = f"""
Извлеки отношения между сущностями
из следующего текста.

Верни ТОЛЬКО JSON array.

Формат:

[
  {{
    "subject": "...",
    "relation": "...",
    "object": "...",
    "evidence": "..."
  }}
]

Не добавляй информацию,
которой нет в тексте.

TEXT:

{text}
"""

    answer = call_ollama([
        {
            "role": "user",
            "content": prompt
        }
    ])
    try:
        return json.loads(answer)

    except json.JSONDecodeError:
        return {
            "raw_output": answer
        }