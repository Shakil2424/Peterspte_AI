
import requests
import json
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def stream_chatbot_response(messages, model="llama2"):
    url = "http://localhost:11434/api/chat"
    payload = {"model": model, "messages": messages}
    
    try:
        logger.info(f"Attempting to connect to Ollama at {url}")
        logger.info(f"Using model: {model}")
        
        with requests.post(url, json=payload, stream=True, timeout=60) as res:
            logger.info(f"Ollama response status: {res.status_code}")
            
            if res.status_code != 200:
                error_msg = f"Ollama server returned status {res.status_code}"
                try:
                    error_detail = res.json()
                    error_msg += f": {error_detail}"
                except:
                    error_msg += f": {res.text}"
                logger.error(error_msg)
                yield f"\n[Ollama Error: {error_msg}]\n"
                yield "\nPlease ensure:\n"
                yield "1. Ollama server is running (ollama serve)\n"
                yield f"2. Model '{model}' is installed (ollama pull {model})\n"
                yield "3. Ollama is accessible at http://localhost:11434\n"
                yield f"4. If you have limited memory, try: ollama pull llama2 (smaller model)\n"
                return
            
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
                        logger.error(f"Parse error: {parse_err}")
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
                    logger.error(f"Parse error at end: {parse_err}")
                    yield f"\n[Parse Error at end: {str(parse_err)}]\n"
                    
    except requests.exceptions.ConnectionError as e:
        error_msg = f"Could not connect to Ollama server at {url}"
        logger.error(error_msg)
        yield f"\n[Connection Error: {error_msg}]\n"
        yield "\nTo fix this:\n"
        yield "1. Start Ollama: ollama serve\n"
        yield "2. Install model: ollama pull llama2 (smaller model)\n"
        yield "3. Check if Ollama is running: ollama list\n"
        
    except requests.exceptions.Timeout as e:
        error_msg = f"Request to Ollama timed out after 60 seconds"
        logger.error(error_msg)
        yield f"\n[Timeout Error: {error_msg}]\n"
        yield "\nTry:\n"
        yield "1. Check if Ollama is responding: curl http://localhost:11434/api/tags\n"
        yield "2. Restart Ollama server\n"
        
    except Exception as request_err:
        error_msg = f"Unexpected error: {str(request_err)}"
        logger.error(error_msg)
        yield f"\n[Request Error: {error_msg}]\n"
