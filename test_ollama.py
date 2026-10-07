import requests

url = "http://localhost:11434/api/generate"

payload = {
    "model": "qwen3:0.6b",
    "prompt": "Ответь только одним словом: работает",
    "stream": False,
    "think": False
}

response = requests.post(
    url,
    json=payload,
    timeout=120
)

response.raise_for_status()

data = response.json()

print("Ответ:", data["response"])
print("Thinking:", data.get("thinking"))