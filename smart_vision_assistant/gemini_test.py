import os
from dotenv import load_dotenv
import importlib.metadata
from PIL import Image

import config

def test_gemini():
    load_dotenv()
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("GEMINI_API_KEY not found in .env")
        return
        
    try:
        sdk_version = importlib.metadata.version("google-genai")
        print(f"SDK Version: google-genai v{sdk_version}")
    except importlib.metadata.PackageNotFoundError:
        print("SDK google-genai is not installed.")
        return

    print(f"Selected Model: {config.GEMINI_MODEL}")

    from google import genai
    client = genai.Client(api_key=api_key)
    
    print("\n--- Listing Models ---")
    models = list(client.models.list())
    for m in models:
        methods = getattr(m, 'supported_actions', []) or getattr(m, 'supported_generation_methods', [])
        print(f"Model ID: {m.name} | Methods: {methods}")

    print("\n--- Sending Text Prompt ---")
    try:
        response = client.models.generate_content(
            model=config.GEMINI_MODEL,
            contents="Reply exactly with OK"
        )
        print("Text Response:")
        print(response.text)
    except Exception as e:
        print("Text Prompt Failed:", e)

    print("\n--- Sending Image Prompt ---")
    try:
        # Create a dummy image
        img = Image.new('RGB', (100, 100), color = 'red')
        response = client.models.generate_content(
            model=config.GEMINI_MODEL,
            contents=[img, "What color is this image? Reply with one word."]
        )
        print("Image Response:")
        print(response.text)
    except Exception as e:
        print("Image Prompt Failed:", e)

if __name__ == "__main__":
    test_gemini()
