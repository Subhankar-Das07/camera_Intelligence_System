import os
from dotenv import load_dotenv

def discover_models():
    load_dotenv()
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("NO API KEY")
        return

    from google import genai
    client = genai.Client(api_key=api_key)
    
    print("AVAILABLE MODELS:")
    models = list(client.models.list())
    for m in models:
        # Check if it supports generateContent
        supported_methods = getattr(m, 'supported_actions', []) or getattr(m, 'supported_generation_methods', [])
        print(f"Model ID: {m.name} | Methods: {supported_methods}")

if __name__ == "__main__":
    discover_models()
