import os
from datetime import datetime, timezone
from typing import Dict, List, Optional
from dotenv import load_dotenv
from pymongo import MongoClient
from pymongo.errors import DuplicateKeyError #type: ignore
import google.generativeai as genai #type: ignore
import requests
from requests.auth import HTTPBasicAuth
import importlib
from sentence_transformers import SentenceTransformer
import numpy as np

# --------------------------------------------------------------------------------
# 1) Load environment variables and configure APIs
# --------------------------------------------------------------------------------
# To run this script, create a .env file with your keys or set them as environment variables
# Example .env file:
# MONGO_URI="your_mongodb_connection_string"
# JIRA_URL="https://your-domain.atlassian.net"
# JIRA_EMAIL="your_jira_email"
# JIRA_API_TOKEN="your_jira_api_token"
# GEMINI_API_KEY="your_gemini_api_key"
# PINECONE_API_KEY="your_pinecone_api_key"
# PINECONE_ENVIRONMENT="your_pinecone_environment"

load_dotenv()

# MongoDB Configuration
MONGO_URI = os.getenv("MONGO_URI")

# JIRA Configuration
JIRA_URL = os.getenv("JIRA_URL")
JIRA_EMAIL = os.getenv("JIRA_EMAIL")
JIRA_API_TOKEN = os.getenv("JIRA_API_TOKEN")
jira_auth = HTTPBasicAuth(JIRA_EMAIL, JIRA_API_TOKEN) if JIRA_EMAIL and JIRA_API_TOKEN else None
jira_headers = {"Accept": "application/json"}

# Gemini Configuration
if os.getenv("GEMINI_API_KEY"):
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    model = genai.GenerativeModel('gemini-1.5-flash')
else:
    model = None
    print("WARNING: GEMINI_API_KEY not found. AI features will be disabled.")


# Pinecone Configuration
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_REGION = os.getenv("PINECONE_ENVIRONMENT", "us-east-1")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "ai-scrum-index")

os.environ["TRANSFORMERS_NO_META_DEVICE_INIT"] = "1"

# Helper to always get a list from embedding
def safe_encode(model, text):
    return model.encode(text, convert_to_tensor=False).tolist()


# --------------------------------------------------------------------------------
# 1.1) Initialize Models and Vector DB
# --------------------------------------------------------------------------------
embedding_model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
EMBEDDING_DIMENSION = 384  # Dimension for 'all-MiniLM-L6-v2' embeddings

# Initialize Pinecone
index = None
if PINECONE_API_KEY:
    from pinecone import Pinecone, ServerlessSpec
    pc = Pinecone(api_key=PINECONE_API_KEY)
    if PINECONE_INDEX_NAME not in [idx.name for idx in pc.list_indexes()]:
        print(f"Index '{PINECONE_INDEX_NAME}' does not exist. Creating...")
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=EMBEDDING_DIMENSION,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region=PINECONE_REGION)
        )
    else:
        print(f"Index '{PINECONE_INDEX_NAME}' already exists.")
    index = pc.Index(PINECONE_INDEX_NAME)
    print("Connected to Pinecone index:", PINECONE_INDEX_NAME)
else:
    print("WARNING: PINECONE_API_KEY not found. Vector search features will be disabled.")


# --------------------------------------------------------------------------------
# 2) MongoDB Setup and Helper Functions
# --------------------------------------------------------------------------------
client = MongoClient(MONGO_URI) if MONGO_URI else None
if client:
    db = client["jira_db"]
    boards_collection = db["boards"]
    sprints_collection = db["sprints"]
    issues_collection = db["issues"]
    users_collection = db["users"]
    conversations_collection = db["conversations"]
    print("Connected to MongoDB.")
else:
    print("WARNING: MONGO_URI not found. MongoDB features will be disabled.")
    db = None # To prevent errors

# Define MongoDB functions (they will do nothing if db is None)
def store_board(board: Dict):
    if not db or not isinstance(board, dict): return
    try:
        boards_collection.update_one({"board_id": board.get('id')}, {"$set": {"name": board.get('name'), "type": board.get('type')}}, upsert=True)
    except DuplicateKeyError: pass

# Other MongoDB functions would follow a similar pattern...

# --------------------------------------------------------------------------------
# 3) JIRA Integration Functions
# --------------------------------------------------------------------------------
def fetch_sprint_details(board_id: int, include_closed: bool = False) -> List[Dict]:
    if not jira_auth: 
        print("Jira credentials not set. Returning mock data.")
        # Return mock data that includes the specified users
        return [{
            'id': 1, 'name': 'Sample Sprint', 'state': 'active', 'goal': 'Demonstrate the bot',
            'issues': [
                {'Key': 'PROJ-1', 'Summary': 'Develop login feature', 'Status': 'In Progress', 'Assignee': {'displayName': 'Amith'}},
                {'Key': 'PROJ-2', 'Summary': 'Design database schema', 'Status': 'To Do', 'Assignee': {'displayName': 'Sachin'}},
                {'Key': 'PROJ-3', 'Summary': 'Set up CI/CD pipeline', 'Status': 'In Progress', 'Assignee': {'displayName': 'Rahul K'}},
                {'Key': 'PROJ-4', 'Summary': 'Write API documentation', 'Status': 'Done', 'Assignee': {'displayName': 'Yuvraj'}},
                {'Key': 'PROJ-5', 'Summary': 'Test user authentication', 'Status': 'To Do', 'Assignee': {'displayName': 'Amith'}},
            ]
        }]
    # ... (Real Jira fetching logic would go here)
    return []


# --------------------------------------------------------------------------------
# 4) AI Scrum Master Class
# --------------------------------------------------------------------------------
class AIScrumMaster:
    def __init__(self, user_id: str):
        self.user_id = user_id
        self.conversation_history = []
        self.current_sprint = None
        self.team_member_details: Dict[str, Dict] = {}
        self.current_member_name: Optional[str] = None # Tracks who the bot is waiting for

    def initialize_sprint_data(self, board_id: int) -> bool:
        """Populates sprint data and creates the user ID map for authentication."""
        sprints = fetch_sprint_details(board_id, include_closed=False)
        if not sprints:
            print("Could not fetch sprint details.")
            return False

        active_sprints = [s for s in sprints if s['state'] == 'active']
        if not active_sprints:
            print("No active sprints found.")
            return False
            
        self.current_sprint = active_sprints[0]
        
        # This map links a user's name (from Jira) to their unique MS Teams ID.
        # In a real app, this would come from a database, not be hardcoded.
        MOCK_TEAMS_ID_MAP = {
            "Amith": "29:1EaTSFcrZkjsH3S6cgT9h88AzHL4AqFSmyE2I7LfhIGBu3QpI9aBXxuB0ChMkDWSRYYQEQuUK6Lc-mGoNzB58FA",
            "Rahul K": "29:1XFcY26AqQUhnEbXtLhwS92rQRZyu_bxLNHQiswlKNKffy9as-xn2rpVmArUCfUtAC8eEaS2O9ZWvM4fOSe4NdA",
            "Sachin": "29:1wIHM0iaGI3IKzW-fHfzzPQ2K5E1MbVjRVoNK2OnodxBhotBQmxgSX1fCB-8laPQbCEEhUzw-PPTyVt_YML3HzQ",
            "Yuvraj": "29:15PQxlsgRCbsfORrMJT0-tjx2986KF1qJCW1b8YwD62ocx37V5R6zd6gNY7jNacSTdZZRwUTiPq7N10zwfDph7Q"
        }

        for issue in self.current_sprint.get('issues', []):
            assignee = issue.get('Assignee')
            if isinstance(assignee, dict):
                display_name = assignee.get('displayName')
                if display_name and display_name not in self.team_member_details:
                    self.team_member_details[display_name] = {
                        'jira_user': assignee,
                        'msteams_id': MOCK_TEAMS_ID_MAP.get(display_name)
                    }
        print("Initialized team members:", list(self.team_member_details.keys()))
        return True
        
    def handle_user_reply(self, responding_teams_id: str, response_text: str) -> Optional[str]:
        """
        Verifies user identity. If correct, processes the response and returns a success message.
        If incorrect, it silently ignores the message and returns None.
        """
        # 1. Ignore unsolicited messages if the bot isn't waiting for a specific user.
        if not self.current_member_name:
            return None

        # 2. Look up the user the bot is waiting for.
        expected_member_info = self.team_member_details.get(self.current_member_name)
        if not expected_member_info:
            print(f"Error: Waiting for {self.current_member_name} but they are not in the team map.")
            return None

        # 3. Get the expected MS Teams ID for that user.
        expected_teams_id = expected_member_info.get('msteams_id')
        if not expected_teams_id:
            # This is a configuration issue (user exists in Jira but not in our ID map).
            # For now, we allow it but log a warning. A stricter system could return None here too.
            print(f"WARNING: No MS Teams ID is mapped for {self.current_member_name}. Processing without verification.")
        
        # 4. **THE AUTHENTICATION CHECK**
        # If the ID from the message does not match the expected ID, return None to ignore.
        elif responding_teams_id != expected_teams_id:
            return None

        # 5. If the check passes, process the response and return a success message.
        self.add_user_response(self.current_member_name, response_text)
        return f"Thanks, {self.current_member_name}! Your update has been recorded."

    def generate_question(self, member_name: str) -> str:
        """
        Generates a question and sets the internal state to wait for that member.
        """
        if not model: return f"Hello {member_name}, what is your update? (AI model not configured)"
        
        self.current_member_name = member_name
        
        # In a real scenario, this would use the Gemini model with full context
        # to generate a more intelligent question.
        # For this example, we'll keep it simple.
        tasks = [
            f"'{issue['Summary']}' ({issue['Status']})"
            for issue in self.current_sprint.get('issues', [])
            if issue.get('Assignee', {}).get('displayName') == member_name
        ]
        
        prompt = f"You are a friendly AI Scrum Master. Ask {member_name} for an update on their tasks. Their tasks are: {', '.join(tasks)}. Keep the question brief and encouraging."
        
        response = model.generate_content(prompt)
        return response.text.strip()

    def add_user_response(self, member_name: str, response: str):
        """Processes and stores a user's response (this is an internal method)."""
        print(f"  [+] Correct user replied. Processing response from {member_name}: '{response}'")
        self.conversation_history.append({
            "role": "user",
            "content": response,
            "member_name": member_name,
            "timestamp": datetime.now(timezone.utc)
        })
        # In a real app, this would also trigger analysis and storage in Pinecone/MongoDB.

if __name__ == '__main__':
    print("--- Running AI Scrum Master Authentication Logic Demo ---")

    # 1. Initialize the bot and load data for a sample project board.
    # We use a mock user ID for the bot itself.
    scrum_bot = AIScrumMaster(user_id="ai_scrum_master_9000")
    
    # The `initialize_sprint_data` method populates the user map from mock Jira data.
    # In your real app, this would connect to Jira.
    scrum_bot.initialize_sprint_data(board_id=1) 
    print("-" * 50)

    # 2. The stand-up begins. The bot decides to ask Amith a question first.
    question_to_amith = scrum_bot.generate_question("Amith")
    print(f"➡️ Bot asks Amith:\n'{question_to_amith}'")
    print(f"   (Internal state: Bot is now waiting for '{scrum_bot.current_member_name}')")
    print("-" * 50)

    # 3. **Scenario: The WRONG user (Sachin) tries to reply first.**
    print("🔴 A message is received from Sachin (the WRONG user)...")
    sachins_teams_id = "29:1wIHM0iaGI3IKzW-fHfzzPQ2K5E1MbVjRVoNK2OnodxBhotBQmxgSX1fCB-8laPQbCEEhUzw-PPTyVt_YML3HzQ"
    sachins_reply_text = "I finished the database schema design."
    
    # The bot handles the reply.
    response_from_bot = scrum_bot.handle_user_reply(sachins_teams_id, sachins_reply_text)
    
    # Your application checks the response.
    if response_from_bot:
        # This code will NOT run.
        print(f"❌ ERROR: Bot should have ignored this message but wants to reply: {response_from_bot}")
    else:
        # This code WILL run.
        print("✅ SUCCESS: Bot returned None and silently ignored the message.")
        print(f"   (Internal state: Bot is still waiting for '{scrum_bot.current_member_name}')")
    print("-" * 50)

    # 4. **Scenario: The CORRECT user (Amith) now replies.**
    print("🟢 A message is received from Amith (the CORRECT user)...")
    amiths_teams_id = "29:1EaTSFcrZkjsH3S6cgT9h88AzHL4AqFSmyE2I7LfhIGBu3QpI9aBXxuB0ChMkDWSRYYQEQuUK6Lc-mGoNzB58FA"
    amiths_reply_text = "I've started work on the login feature, no blockers so far."

    # The bot handles the reply.
    response_from_bot = scrum_bot.handle_user_reply(amiths_teams_id, amiths_reply_text)

    # Your application checks the response again.
    if response_from_bot:
        # This code WILL run.
        print("✅ SUCCESS: Bot processed the message and returned a confirmation.")
        print(f"➡️ Bot should reply to Amith:\n'{response_from_bot}'")
        
        # After a successful interaction, your application would clear the state
        # to prepare for the next person.
        scrum_bot.current_member_name = None
        print(f"   (Internal state: Bot is no longer waiting for anyone and can move on.)")
    else:
        # This code will NOT run.
        print("❌ ERROR: Bot should have processed this message.")
    print("-" * 50)