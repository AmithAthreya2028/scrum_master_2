import os
from datetime import datetime, timezone
from typing import Dict, List, Optional
from dotenv import load_dotenv
from pymongo import MongoClient
from pymongo.errors import DuplicateKeyError #type: ignore
import google.generativeai as genai #type: ignore
import requests
from requests.auth import HTTPBasicAuth
# import re
# import json
# import pandas as pd

# Pinecone import compatible with both classic and new SDK
import importlib
pinecone_version = None
Pinecone = None
pinecone = None
IndexSpec = None
try:
    pinecone_mod = importlib.import_module("pinecone")
    if hasattr(pinecone_mod, "Pinecone"):
        Pinecone = getattr(pinecone_mod, "Pinecone")
        IndexSpec = getattr(pinecone_mod, "IndexSpec", None)
        pinecone_version = 3
    else:
        pinecone = pinecone_mod
        pinecone_version = 2
except Exception :
    Pinecone = None
    pinecone = None
    pinecone_version = None
    IndexSpec = None

from sentence_transformers import SentenceTransformer
import numpy as np
os.environ["TRANSFORMERS_NO_META_DEVICE_INIT"] = "1"

# Helper to always get a list from embedding
def safe_encode(model, text):
    return model.encode(text, convert_to_tensor=False).tolist()


# --------------------------------------------------------------------------------
# 1) Load environment variables and configure APIs
# --------------------------------------------------------------------------------
import os
from dotenv import load_dotenv

# Print the current GEMINI_API_KEY at startup for debugging
print("GEMINI_API_KEY at startup:", os.getenv("GEMINI_API_KEY"))

# Only load .env if GEMINI_API_KEY is not already set in the environment
if not os.getenv("GEMINI_API_KEY"):
    load_dotenv()

# MongoDB Configuration
MONGO_URI = os.getenv("MONGO_URI")

# JIRA Configuration
JIRA_URL = os.getenv("JIRA_URL")
JIRA_EMAIL = os.getenv("JIRA_EMAIL")
JIRA_API_TOKEN = os.getenv("JIRA_API_TOKEN")
jira_auth = HTTPBasicAuth(JIRA_EMAIL, JIRA_API_TOKEN)#type: ignore
jira_headers = {"Accept": "application/json"}

# Gemini Configuration
print("Gemini API Key (debug):", os.getenv("GEMINI_API_KEY"))  # Debug print for verification
genai.configure(api_key=os.getenv("GEMINI_API_KEY")) #type: ignore
model = genai.GenerativeModel('gemini-1.5-flash')#type: ignore

# Pinecone Configuration
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_REGION = os.getenv("PINECONE_ENVIRONMENT") or "us-east-1"
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "ai-scrum-index")

# --------------------------------------------------------------------------------
# 1.1) Initialize Pinecone and Embedding Model
# --------------------------------------------------------------------------------
embedding_model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
EMBEDDING_DIMENSION = 384  # Dimension for 'all-MiniLM-L6-v2' embeddings

# Pinecone v3 initialization
from pinecone import Pinecone, ServerlessSpec

print("Pinecone API Key:", PINECONE_API_KEY)
print("Pinecone Region:", PINECONE_REGION)
pc = Pinecone(api_key=PINECONE_API_KEY)

# Check if the index exists, and create it if not
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

# --------------------------------------------------------------------------------
# 2) MongoDB Setup
# --------------------------------------------------------------------------------
client = MongoClient(MONGO_URI)
db = client["jira_db"]
boards_collection = db["boards"]
sprints_collection = db["sprints"]
issues_collection = db["issues"]
users_collection = db["users"]
conversations_collection = db["conversations"]

# --------------------------------------------------------------------------------
# 2.1) MongoDB Helper Functions
# --------------------------------------------------------------------------------
def store_board(board: Dict):
    """Store a Jira board document into MongoDB."""
    if not isinstance(board, dict):
        return
    board_doc = {
        "board_id": board.get('id'),
        "name": board.get('name'),
        "type": board.get('type'),
        "created_at": datetime.now(timezone.utc),  # Use UTC for consistency
    }
    try:
        boards_collection.insert_one(board_doc)
    except DuplicateKeyError:
        print(f"Board with id {board.get('id')} already exists.")

def store_sprint(sprint: Dict, board_id: int):
    """Store a sprint document into MongoDB."""
    if not isinstance(sprint, dict):
        return
    sprint_doc = {
        "sprint_id": sprint.get('id'),
        "board_id": board_id,
        "name": sprint.get('name'),
        "state": sprint.get('state'),
        "start_date": sprint.get('startDate'),
        "end_date": sprint.get('endDate'),
        "goal": sprint.get('goal', 'No goal set'),
        "issues": [issue.get('Key') if isinstance(issue, dict) else str(issue) for issue in sprint.get('issues', [])]
    }
    try:
        sprints_collection.insert_one(sprint_doc)
    except DuplicateKeyError:
        print(f"Sprint with id {sprint.get('id')} already exists.")

def store_issue(issue: Dict, board_id: int, sprint_id: int):
    """Store an issue document into MongoDB."""
    if not isinstance(issue, dict):
        return
    issue_doc = {
        "issue_id": issue.get('Key'),
        "board_id": board_id,
        "sprint_id": sprint_id,
        "summary": issue.get('Summary'),
        "status": issue.get('Status'),
        "assignee": issue.get('Assignee'),
        "story_points": issue.get('story_points', None),
        "created_at": issue.get('Created'),
        "updated_at": issue.get('Updated')
    }
    try:
        issues_collection.insert_one(issue_doc)
    except DuplicateKeyError:
        print(f"Issue with id {issue.get('Key')} already exists.")

def store_user(user_id: str, display_name: str, jira_account_id: Optional[str] = None, msteams_id: Optional[str] = None):
    """Store or update a user document in MongoDB with IDs from different systems."""
    user_doc = {
        "user_id": user_id,
        "display_name": display_name,
        "created_at": datetime.now(timezone.utc)
    }
    update_set = {"$set": user_doc}
    if jira_account_id:
        update_set["$set"]["jira_account_id"] = jira_account_id
    if msteams_id:
        update_set["$set"]["msteams_id"] = msteams_id
        
    users_collection.update_one(
        {"user_id": user_id},
        update_set,
        upsert=True
    )

def get_last_selected_board(user_id: str) -> Optional[int]:
    """Retrieve the last selected board for a user from MongoDB."""
    doc = users_collection.find_one({"user_id": user_id})
    if doc and "last_board_id" in doc:
        return doc["last_board_id"]
    return None

def set_last_selected_board(user_id: str, board_id: int):
    """Store the last selected board for a user in MongoDB."""
    users_collection.update_one(
        {"user_id": user_id},
        {"$set": {"last_board_id": board_id}},
        upsert=True
    )

def store_conversation(conversation_doc: dict):
    """Store a conversation document into MongoDB."""
    conversation_doc["date"] = datetime.now(timezone.utc)
    conversations_collection.insert_one(conversation_doc)

def get_previous_standups(user_id: str, limit=5):
    """Retrieve recent standup documents from MongoDB for a specific user."""
    cursor = conversations_collection.find({"user_id": user_id}).sort("date", -1).limit(limit)
    return list(cursor)

# --------------------------------------------------------------------------------
# 3) JIRA Integration Functions
# --------------------------------------------------------------------------------
def extract_content_from_adf(content):
    """Extract plain text from Atlassian Document Format (ADF)."""
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        if 'text' in content:
            return content['text']
        if 'content' in content:
            return ' '.join(extract_content_from_adf(c) for c in content['content'])
    if isinstance(content, list):
        return ' '.join(extract_content_from_adf(c) for c in content)
    return ''

def get_field_value(issue: Dict, field_name: str) -> str:
    """Extract specific field values with proper fallback."""
    if not isinstance(issue, dict):
        return "Not available"
    fields = issue.get('fields', {})
    if not isinstance(fields, dict):
        return "Not available"
    if field_name == 'description':
        content = fields.get('description')
        return extract_content_from_adf(content) if content else "No description available"
    if field_name == 'assignee':
        assignee = fields.get('assignee')
        return assignee.get('displayName') if isinstance(assignee, dict) else "Unassigned"
    if field_name == 'status':
        status = fields.get('status')
        return status.get('name') if isinstance(status, dict) else "Unknown"
    return str(fields.get(field_name, "Not available"))

def get_issue_details(issue: Dict) -> Dict:
    """Return a dictionary with key details about an issue."""
    if not isinstance(issue, dict):
        return {}
    fields = issue.get('fields', {})
    if not isinstance(fields, dict):
        return {
            'Key': issue.get('key'),
            'Summary': "Invalid fields",
            'Status': "Invalid fields",
            'Assignee': None,
            'Reporter': "Invalid fields",
            'Priority': "Invalid fields",
            'Issue Type': "Invalid fields",
            'Created': "Invalid fields",
            'Updated': "Invalid fields",
            'Description': "Invalid fields"
        }
    
    priority = fields.get('priority')
    issuetype = fields.get('issuetype')

    return {
        'Key': issue.get('key'),
        'Summary': get_field_value(issue, 'summary'),
        'Status': get_field_value(issue, 'status'),
        'Assignee': fields.get('assignee'),  # <-- preserve full object
        'Reporter': get_field_value(issue, 'reporter'),
        'Priority': priority.get('name', 'Not set') if isinstance(priority, dict) else 'Not set',
        'Issue Type': issuetype.get('name', 'Unknown') if isinstance(issuetype, dict) else 'Unknown',
        'Created': fields.get('created', 'Unknown'),
        'Updated': fields.get('updated', 'Unknown'),
        'Description': get_field_value(issue, 'description')
    }

def get_boards() -> List[Dict]:
    """Fetch all available Scrum boards from JIRA."""
    url = f"{JIRA_URL}/rest/agile/1.0/board"
    response = requests.get(url, headers=jira_headers, auth=jira_auth)
    if response.status_code == 200:
        boards = response.json().get('values', [])
        for board in boards:
            store_board(board)
        return boards
    else:
        print(f"Error fetching boards: {response.status_code} {response.text}")
        return []

def fetch_sprint_details(board_id: int, include_closed: bool = False) -> List[Dict]:
    """Fetch sprints and their issues for the given board."""
    url = f"{JIRA_URL}/rest/agile/1.0/board/{board_id}/sprint"
    response = requests.get(url, headers=jira_headers, auth=jira_auth)
    sprints_list = []
    if response.status_code == 200:
        for sprint in response.json().get('values', []):
            sprint_id = sprint['id']
            issues_url = f"{JIRA_URL}/rest/agile/1.0/sprint/{sprint_id}/issue"
            issues_response = requests.get(issues_url, headers=jira_headers, auth=jira_auth)
            issues = []
            if issues_response.status_code == 200:
                issues = [get_issue_details(issue) for issue in issues_response.json().get('issues', [])]
            sprint_data = {
                'id': sprint_id,
                'name': sprint.get('name', 'N/A'),
                'state': sprint.get('state', 'N/A'),
                'start_date': sprint.get('startDate', 'N/A'),
                'end_date': sprint.get('endDate', 'N/A'),
                'goal': sprint.get('goal', 'No goal set'),
                'issues': issues
            }
            store_sprint(sprint_data, board_id)
            for issue in issues:
                store_issue(issue, board_id, sprint_id)
            sprints_list.append(sprint_data)
        return sprints_list
    else:
        print(f"Error fetching sprints: {response.status_code} {response.text}")
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
        self.blockers = []
        self.action_items = []
        self.context_cache = {}
        self.current_member_name: Optional[str] = None
        self.team_members = []  # <-- Added to prevent attribute error

        # Initialize with a system prompt
        self.system_prompt = (
            "You are an AI Scrum Master named AgileBot. You greet team members warmly, "
            "ask about their tasks, blockers, and updates in a friendly, empathetic, "
            "ask limited questions to avoid overwhelming the team and finish the standup early "
            "DO NOT ask redundant questions , if you are not satisfied with the answer, just move to the next important question because time is very limited"
            "If you do not get a clear answer to your question, ask a followup question and move onto the next question."
            "Do no repeat any questions and end the standup as early as possible to conserve time as standups typically take 5-7 minutes per user"
            "and concise way. Always maintain a helpful and professional tone and use bullet points when helpful."
        )
        self.conversation_history.append({
            "role": "system",
            "content": self.system_prompt,
            "timestamp": datetime.now(timezone.utc)
        })

        # Load previous standups
        previous_standups = get_previous_standups(self.user_id, limit=3)
        for doc in reversed(previous_standups):
            summary = doc.get("summary")
            if summary:
                self.conversation_history.append({
                    "role": "system",
                    "content": f"[Previous Standup Summary]\n{summary}",
                    "timestamp": doc.get("date")
                })

    def initialize_sprint_data(self, board_id: int):
        """Initialize sprint data, and build a map of team members with their Jira and Teams IDs."""
        sprints = fetch_sprint_details(board_id, include_closed=False)
        if sprints:
            active_sprints = [s for s in sprints if s['state'] == 'active']
            if active_sprints:
                self.current_sprint = active_sprints[0]
                
                ### MODIFICATION: Using the MS Teams IDs you provided.
                # In a real app, this map would be built by querying your MongoDB user collection.
                MOCK_TEAMS_ID_MAP = {
                    "Amith": "29:1EaTSFcrZkjsH3S6cgT9h88AzHL4AqFSmyE2I7LfhIGBu3QpI9aBXxuB0ChMkDWSRYYQEQuUK6Lc-mGoNzB58FA",
                    "Rahul K": "29:1XFcY26AqQUhnEbXtLhwS92rQRZyu_bxLNHQiswlKNKffy9as-xn2rpVmArUCfUtAC8eEaS2O9ZWvM4fOSe4NdA",
                    "Sachin": "29:1wIHM0iaGI3IKzW-fHfzzPQ2K5E1MbVjRVoNK2OnodxBhotBQmxgSX1fCB-8laPQbCEEhUzw-PPTyVt_YML3HzQ",
                    "Yuvraj": "29:15PQxlsgRCbsfORrMJT0-tjx2986KF1qJCW1b8YwD62ocx37V5R6zd6gNY7jNacSTdZZRwUTiPq7N10zwfDph7Q"
                    # NOTE: Ensure the keys here (e.g., "Amith") exactly match the 'displayName' in Jira.
                }

                for issue in self.current_sprint['issues']:
                    assignee = issue.get('Assignee')
                    if isinstance(assignee, dict):
                        display_name = assignee.get('displayName')
                        if display_name and display_name not in self.team_member_details:
                            self.team_member_details[display_name] = {
                                'jira_user': assignee,
                                'msteams_id': MOCK_TEAMS_ID_MAP.get(display_name)
                            }

                if not self.team_member_details:
                    print("Warning: No assigned team members found in the active sprint.")
                return True
        return False
    
    def handle_user_reply(self, responding_teams_id: str, response_text: str) -> str:
        """
        Verifies the user identity before processing their response.
        
        Args:
            responding_teams_id: The `from.id` of the user who sent the message in MS Teams.
            response_text: The `text` content of the message.
            
        Returns:
            A string indicating success or an error message.
        """
        if not self.current_member_name:
            return "Thanks for the message, but I'm not currently waiting for a response from anyone. Let's start the stand-up."

        expected_member_info = self.team_member_details.get(self.current_member_name)
        if not expected_member_info:
            return f"Error: I was expecting a reply from {self.current_member_name}, but they are not in my list of team members."
            
        expected_teams_id = expected_member_info.get('msteams_id')
        
        if not expected_teams_id:
            print(f"WARNING: No MS Teams ID is mapped for {self.current_member_name}. Skipping user verification for this turn.")
        
        elif responding_teams_id != expected_teams_id:
            responding_user_name = "an unknown user"
            for name, details in self.team_member_details.items():
                if details.get('msteams_id') == responding_teams_id:
                    responding_user_name = name
                    break
            
            error_message = (
                f"Hold on! I was expecting a response from **{self.current_member_name}**, "
                f"but this reply came from **{responding_user_name}**. "
                "Please wait for your turn."
            )
            return error_message

        self.add_user_response(self.current_member_name, response_text)
        return f"Thanks, {self.current_member_name}! I've recorded your update."


    def get_member_tasks(self, member_name: str) -> List[Dict]:
        """Get active tasks for a team member from the current sprint."""
        if not self.current_sprint:
            return []
        return [
            issue for issue in self.current_sprint['issues']
            if isinstance(issue, dict) and isinstance(issue.get('Assignee'), dict) and issue.get('Assignee', {}).get('displayName') == member_name
        ]

    def build_tasks_context(self, member_name: str) -> str:
        """Build context string for member's tasks."""
        tasks = self.get_member_tasks(member_name)
        if not tasks:
            return "No tasks assigned currently."
        return "\n".join([
            f"- {task.get('Key', 'N/A')}: {task.get('Summary', 'N/A')} (Status: {task.get('Status', 'N/A')})"
            for task in tasks if isinstance(task, dict)
        ])

    def get_mongo_context(self, member_name: str) -> str:
        docs = get_previous_standups(self.user_id, limit=3)
        summaries = [doc.get("summary") for doc in docs if doc.get("summary")]
        if summaries:
            return "\nRecent Standup Summaries:\n" + "\n".join(f"- {s}" for s in summaries)
        return "No historical updates available."

    def get_contextual_history(self, member_name: str) -> str:
        """Get relevant historical context for the team member."""
        if member_name in self.context_cache:
            return self.context_cache[member_name]
        query = f"{member_name}'s recent updates"
        contexts = self.fetch_relevant_context(query)
        context_str = "\nRelevant History:\n" + "\n".join([f"- {ctx['text']}" for ctx in contexts])
        self.context_cache[member_name] = context_str
        return context_str

    def fetch_relevant_context(self, query: str, top_k: int = 5) -> list:
        if not index:
            return []
        try:
            vector = safe_encode(embedding_model, query)
            results = index.query(
                vector=vector,
                top_k=top_k,
                include_metadata=True
            )
            return [match.metadata for match in results.matches]#type: ignore
        except Exception as e:
            print(f"Failed to fetch relevant context from Pinecone: {str(e)}")
            return []



    def generate_question(self, member_name: str, step: int) -> str:
        """
        Generate the next appropriate question for the user and sets the context
        to expect a reply from this specific user.
        """
        self.current_member_name = member_name

        member_history = []
        for msg in self.conversation_history:
            if msg.get("member_name") == member_name:
                member_history.append({"role": msg["role"], "content": msg["content"]})

        qa_history = ""
        for msg in member_history:
            if msg["role"] == "assistant":
                qa_history += f"Assistant asked: {msg['content']}\n"
            elif msg["role"] == "user":
                qa_history += f"{member_name} replied: {msg['content']}\n"
        if not qa_history:
            qa_history = f"No prior conversation history for {member_name} in this standup. This is the first question for {member_name}."

        previous_standups = get_previous_standups(self.user_id, limit=3)
        previous_summaries = [doc.get("summary") for doc in previous_standups if doc.get("summary")]
        previous_context = "\n".join(f"- {summary}" for summary in previous_summaries) if previous_summaries else "No previous standup summaries available."

        tasks_context = self.build_tasks_context(member_name)
        member_tasks = self.get_member_tasks(member_name)

        cross_user_contexts = []
        for task in member_tasks:
            task_key = task.get('Key')
            if task_key:
                cross_context = self.fetch_cross_user_context(task_key, exclude_user_id=self.user_id)
                if cross_context:
                    cross_user_contexts.append({
                        "task_key": task_key,
                        "context": cross_context
                    })

        cross_user_context_str = ""
        for item in cross_user_contexts:
            cross_user_updates = "\n".join(
                f"- {ctx.get('member_name', 'Unknown')}: {ctx.get('text', '')}"
                for ctx in item["context"]
            )
            cross_user_context_str += f"\nOther team members' updates for task {item['task_key']}:\n{cross_user_updates}\n"

        scrum_questions = [
            "What did you work on since the last standup?",
            "What are you planning to work on today?",
            "Are there any blockers or impediments in your way?",
            "Is there anything else you'd like to share with the team?"
        ]

        prompt = f"""
    You are an AI Scrum Master conducting a standup with {member_name}.

    Here are the tasks assigned to {member_name} in the current sprint:
    {tasks_context}

    {cross_user_context_str}

    Here is the conversation so far in the current standup:
    {qa_history}

    Here are summaries from previous standups for {member_name}:
    {previous_context}

    Your task:
    - For each JIRA task listed above, ask the user for a status update, blockers, and next steps, one task at a time.
    - Reference what other team members have said about the same task if available.
    - Do NOT finish the standup until all tasks have been discussed, unless the user explicitly says they have nothing more to add for all tasks.
    - Reference the JIRA tasks above directly in your questions (use their IDs and summaries).
    - Do NOT ask about topics that {member_name} has already answered or declined (e.g., said 'no', 'nothing', or similar).
    - If a topic has been covered, move on to the next relevant Scrum question or task.
    - Only ask a follow-up if clarification is genuinely needed and has not already been declined.
    - The standard Scrum questions are: {', '.join(scrum_questions)}

    Now, generate the next appropriate question for {member_name}, or move to the next team member only after all tasks have been discussed or the user has nothing more to add.
    """

        refined_question = model.generate_content(prompt).text.strip()
        if not refined_question:
            return "Thank you, all questions have been answered!"
        return refined_question

    def add_user_response(self, member_name: str, response: str):
        """Processes and stores a user's response after it has been verified."""
        self.conversation_history.append({
            "role": "user",
            "content": response,
            "member_name": member_name,
            "timestamp": datetime.now(timezone.utc)
        })
        
        analysis_prompt = f"""
Analyze this response from {member_name}:
---
{response}
---
Provide:
1. Key points (tasks done or in progress)
2. Any blockers/impediments noted
3. Suggested action items/follow-ups
4. Which standard scrum questions (tasks done, blockers, plans, anything else) does this response answer? Respond with a JSON object with keys: "tasks_done", "blockers", "plans", "other", and values true/false.
Please format your answer as a bullet list, and include the JSON object at the end.
"""
        analysis_result = model.generate_content(analysis_prompt).text.strip()
        analysis_message = f"[Internal Analysis]\n{analysis_result}"
        self.conversation_history.append({
            "role": "assistant",
            "content": analysis_message,
            "timestamp": datetime.now(timezone.utc)
        })
        self.store_context_in_pinecone(member_name, response, analysis_result)

    def add_assistant_response(self, response: str, member_name: str):
        self.conversation_history.append({
            "role": "assistant",
            "content": response,
            "member_name": member_name,
            "timestamp": datetime.now(timezone.utc)
        })

    def check_response_completeness(self, member_name: str, response: str) -> bool:
        normalized = response.strip().lower()
        if normalized in ["nothing", "nothing thank you", "no", "none"]:
            return True

        prompt = f"""
You are an AI Scrum Master. Analyze the following standup response from {member_name}:
---
{response}
---
Consider the response 'Complete' if the user gives any reasonable update, even if it is brief, informal, or non-technical (e.g., 'I'm working on it', 'All good', 'No blockers', 'Progressing', etc.).
Only mark as 'Incomplete' if the response is completely missing, off-topic, or does not address the question at all.
Answer with a single word: "Complete" or "Incomplete", followed by a brief explanation.
"""
        result = model.generate_content(prompt).text.strip()
        print("Completeness Analysis:", result)
        if result.lower().startswith("complete"):
            return True
        return False

    def generate_summary(self) -> str:
        recent_history = self.conversation_history
        participants = set()
        for msg in recent_history:
            if isinstance(msg, dict) and msg.get("role") == "user" and msg.get("member_name"):
                participants.add(msg["member_name"])
        
        all_team_members = list(self.team_member_details.keys())

        summary_prompt = f"""
Summarize the following standup conversation:
---
{recent_history}
---
Team members expected: {', '.join(all_team_members)}
Team members who participated: {', '.join(participants)}
For each participant, summarize their updates (even if brief). For team members who did not participate, note "no update provided."
Include:
- Key updates per team member
- Identified blockers
- Action items/follow-ups
- Overall sprint progress
Format the summary in markdown.
"""
        return model.generate_content(summary_prompt).text.strip()

    def store_context_in_pinecone(self, member_name: str, response: str, analysis_result: str):
        if not index:
            return
        task_key = None
        if self.current_sprint:
            member_tasks = self.get_member_tasks(member_name)
            if member_tasks and isinstance(member_tasks[0], dict):
                task_key = member_tasks[0].get('Key')

        text = f"{member_name}'s response: {response}\nAnalysis: {analysis_result}"
        vector = safe_encode(embedding_model, text)
        vector_id = f"{self.user_id}-{datetime.now(timezone.utc).timestamp()}"
        metadata = {
            "user_id": self.user_id,
            "member_name": member_name,
            "text": text,
            "source": "standup_conversation",
            "timestamp": datetime.now(timezone.utc).timestamp(),
            "task_key": task_key
        }
        try:
            index.upsert([(vector_id, vector, metadata)])
            self.context_cache.pop(member_name, None)
        except Exception as e:
            print(f"Failed to store context in Pinecone: {str(e)}")

    def fetch_cross_user_context(self, task_key: Optional[str], exclude_user_id: Optional[str] = None, top_k: int = 5) -> list:
        if not index or not task_key:
            return []
        try:
            filter_dict = {"task_key": {"$eq": task_key}}
            if exclude_user_id:
                filter_dict["user_id"] = {"$ne": exclude_user_id}
            results = index.query(
                vector=np.zeros(EMBEDDING_DIMENSION).tolist(),
                top_k=top_k,
                include_metadata=True,
                filter=filter_dict #type: ignore
            )
            return [match.metadata for match in results.matches]#type: ignore
        except Exception as e:
            print(f"Failed to fetch cross-user context from Pinecone: {str(e)}")
            return []

    def fetch_semantic_cross_user_context(self, task_description: str, exclude_user_id: Optional[str] = None, top_k: int = 5) -> list:
        if not index or not task_description:
            return []
        try:
            vector = safe_encode(embedding_model, task_description)
            filter_dict = {}
            if exclude_user_id:
                filter_dict["user_id"] = {"$ne": exclude_user_id}
            results = index.query(
                vector=vector,
                top_k=top_k,
                include_metadata=True,
                filter=filter_dict
            )
            matches = getattr(results, "matches", [])
            threshold = 0.7
            return [
                match.metadata for match in matches
                if hasattr(match, "metadata") and getattr(match, "score", 0) >= threshold
            ]
        except Exception as e:
            print(f"Failed to fetch semantic cross-user context: {str(e)}")
            return []