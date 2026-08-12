import os
from dotenv import load_dotenv
from google import genai

def test_api():
    load_dotenv()
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("Error: GEMINI_API_KEY not found in .env")
        return
    
    try:
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model="gemini-2.5-flash", 
            contents="Say 'API is working!'"
        )
        print("Success! Gemini API responded with:")
        print(response.text)
    except Exception as e:
        print("Failed to call API:", e)

if __name__ == "__main__":
    test_api()
