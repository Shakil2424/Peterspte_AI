
import requests
import json

def stream_chatbot_response(messages, model="llama3"):
    url = "http://localhost:11434/api/chat"
    payload = {"model": model, "messages": messages}
    
    try:
        with requests.post(url, json=payload, stream=True, timeout=60) as res:
            res.raise_for_status()
            buffer = b""
            for chunk in res.iter_content(chunk_size=1024):
                if not chunk:
                    continue
                buffer += chunk
                while b"\n" in buffer:
                    line, buffer = buffer.split(b"\n", 1)
                    if not line.strip():
                        continue
                    try:
                        decoded_line = line.decode('utf-8')
                        data = json.loads(decoded_line)
                        content = data.get("message", {}).get("content", "")
                        if content:
                            yield content
                    except Exception as parse_err:
                        yield f"\n[Parse Error: {str(parse_err)}]\n"
            # process remaining buffered data
            if buffer:
                try:
                    decoded_line = buffer.decode('utf-8')
                    data = json.loads(decoded_line)
                    content = data.get("message", {}).get("content", "")
                    if content:
                        yield content
                except Exception as parse_err:
                    yield f"\n[Parse Error at end: {str(parse_err)}]\n"
    except Exception as request_err:
        yield f"\n[Request Error: {str(request_err)}]\n"
