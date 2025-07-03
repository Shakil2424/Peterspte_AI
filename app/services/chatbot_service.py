import requests
import json

def stream_chatbot_response(messages, model="llama3"):
    url = "http://localhost:11434/api/chat"
    payload = {
        "model": model,
        "messages": messages
    }
    try:
        with requests.post(url, json=payload, stream=True) as res:
            res.raise_for_status()
            for line in res.iter_lines():
                if line:
                    decoded_line = line.decode('utf-8')
                    data = json.loads(decoded_line)
                    content = data.get("message", {}).get("content", "")
                    if content:
                        yield content
    except Exception as e:
        yield f"\n[Error: {str(e)}]\n" 