import os
from dotenv import load_dotenv

def check_gemini_api_key():
    try:
        import google.generativeai as genai
    except ImportError:
        print("google-generativeai package not installed. Please install it with 'pip install google-generativeai'.")
        return

    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("GEMINI_API_KEY not found in environment variables.")
        return

    genai.configure(api_key=api_key)
    try:
        # Try a simple model info call
        model = genai.GenerativeModel('gemini-2.5-flash')
        response = model.generate_content("Say hello!")
        print("API key is valid. Gemini response:", response.text.strip())
    except Exception as e:
        print("API key check failed. Error:", str(e))

if __name__ == "__main__":
    check_gemini_api_key()
