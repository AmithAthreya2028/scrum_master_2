import os
from dotenv import load_dotenv
import requests

def check_gemini_api_key():
    try:
        import google.generativeai as genai
    except ImportError:
        print("google-generativeai package not installed. Please install it with 'pip install google-generativeai'.")
        return

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("GEMINI_API_KEY not found in environment variables.")
        return

    genai.configure(api_key=api_key)
    try:
        model = genai.GenerativeModel('gemini-2.5-flash')
        response = model.generate_content("Say hello!")
        print("Gemini API key is valid. Gemini response:",api_key, response.text.strip())
    except Exception as e:
        print("Gemini API key check failed. Error:", str(e))

def check_jira_api():
    jira_url = os.getenv("JIRA_URL")
    jira_email = os.getenv("JIRA_EMAIL")
    jira_api_token = os.getenv("JIRA_API_TOKEN")
    if not all([jira_url, jira_email, jira_api_token]):
        print("JIRA_URL, JIRA_EMAIL, or JIRA_API_TOKEN not found in environment variables.")
        return
    try:
        resp = requests.get(
            f"{jira_url}/rest/agile/1.0/board",
            auth=(jira_email, jira_api_token),
            headers={"Accept": "application/json"}
        )
        print(f"JIRA API status: {resp.status_code}")
        if resp.status_code == 200:
            print("JIRA API key is valid.")
        else:
            print(f"JIRA API key check failed. Response: {resp.text}")
    except Exception as e:
        print("JIRA API key check failed. Error:", str(e))

def check_pinecone_api():
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    pinecone_env = os.getenv("PINECONE_ENVIRONMENT")
    if not pinecone_api_key:
        print("PINECONE_API_KEY not found in environment variables.")
        return
    try:
        import pinecone
        # Pinecone v3+ usage
        if hasattr(pinecone, "Pinecone"):
            pc = pinecone.Pinecone(api_key=pinecone_api_key)
            indexes = pc.list_indexes()
            print("Pinecone API key is valid. Indexes:", [idx.name for idx in indexes])
        else:
            # fallback for classic SDK
            pinecone.init(api_key=pinecone_api_key, environment=pinecone_env or "us-east-1")
            indexes = pinecone.list_indexes()
            print("Pinecone API key is valid. Indexes:", indexes)
    except Exception as e:
        print("Pinecone API key check failed. Error:", str(e))

def check_mongo_uri():
    mongo_uri = os.getenv("MONGO_URI")
    if not mongo_uri:
        print("MONGO_URI not found in environment variables.")
        return
    try:
        from pymongo import MongoClient
        client = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)
        client.server_info()  # Will throw if cannot connect
        print("MongoDB URI is valid and connection succeeded.")
    except Exception as e:
        print("MongoDB URI check failed. Error:", str(e))

def check_ms_teams_app():
    app_id = os.getenv("MICROSOFT_APP_ID")
    app_password = os.getenv("MICROSOFT_APP_PASSWORD")
    if not app_id or not app_password:
        print("MICROSOFT_APP_ID or MICROSOFT_APP_PASSWORD not found in environment variables.")
        return
    # Cannot fully check validity without making a real OAuth request, but can check presence
    print("Microsoft Teams App ID and Password are present.")

def check_all_apis():
    print("Checking Gemini API Key...")
    check_gemini_api_key()
    print("\nChecking JIRA API...")
    check_jira_api()
    print("\nChecking Pinecone API...")
    check_pinecone_api()
    print("\nChecking MongoDB URI...")
    check_mongo_uri()
    print("\nChecking Microsoft Teams App credentials...")
    check_ms_teams_app()

if __name__ == "__main__":
    load_dotenv()
    check_all_apis()
