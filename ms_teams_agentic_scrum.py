
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
model = genai.GenerativeModel('gemini-2.5-flash')#type: ignore

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
    sprint_doc = {
        "sprint_id": sprint.get('id'),
        "board_id": board_id,
        "name": sprint.get('name'),
        "state": sprint.get('state'),
        "start_date": sprint.get('startDate'),
        "end_date": sprint.get('endDate'),
        "goal": sprint.get('goal', 'No goal set'),
        "issues": [issue.get('Key') for issue in sprint.get('issues', [])]
    }
    try:
        sprints_collection.insert_one(sprint_doc)
    except DuplicateKeyError:
        print(f"Sprint with id {sprint.get('id')} already exists.")

def store_issue(issue: Dict, board_id: int, sprint_id: int):
    """Store an issue document into MongoDB."""
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

def store_user(user_id: str, display_name: str):
    """Store a user document into MongoDB."""
    user_doc = {
        "user_id": user_id,
        "display_name": display_name,
        "created_at": datetime.now(timezone.utc)
    }
    try:
        users_collection.insert_one(user_doc)
    except DuplicateKeyError:
        print(f"User with id {user_id} already exists.")

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
    fields = issue.get('fields', {})
    if field_name == 'description':
        content = fields.get('description')
        return extract_content_from_adf(content) if content else "No description available"
    if field_name == 'assignee':
        assignee = fields.get('assignee')
        return assignee.get('displayName') if assignee else "Unassigned"
    if field_name == 'status':
        status = fields.get('status')
        return status.get('name') if status else "Unknown"
    return str(fields.get(field_name, "Not available"))

def get_issue_details(issue: Dict) -> Dict:
    """Return a dictionary with key details about an issue."""
    fields = issue.get('fields', {})
    return {
        'Key': issue.get('key'),
        'Summary': get_field_value(issue, 'summary'),
        'Status': get_field_value(issue, 'status'),
        'Assignee': get_field_value(issue, 'assignee'),
        'Reporter': get_field_value(issue, 'reporter'),
        'Priority': fields.get('priority', {}).get('name', 'Not set'),
        'Issue Type': fields.get('issuetype', {}).get('name', 'Unknown'),
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
        self.user_id = user_id  # To track user-specific data
        self.conversation_history = []
        self.current_sprint = None
        self.team_members = set()
        self.blockers = []
        self.action_items = []
        self.context_cache = {}  # For caching contextual history

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
        # Only load summaries from previous standups, not full messages
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
        """Initialize sprint data from JIRA."""
        sprints = fetch_sprint_details(board_id, include_closed=False)
        if sprints:
            active_sprints = [s for s in sprints if s['state'] == 'active']
            if active_sprints:
                self.current_sprint = active_sprints[0]
                for issue in self.current_sprint['issues']:
                    assignee = issue.get('Assignee')
                    if assignee and assignee != "Unassigned":
                        self.team_members.add(assignee)
                        store_user(assignee, assignee)
                return True
        if not self.team_members:
            print("Warning: No team members found in the active sprint. Standup will skip to summary.")
        return False

    def get_member_tasks(self, member_name: str) -> List[Dict]:
        """Get active tasks for a team member from the current sprint."""
        if not self.current_sprint:
            return []
        return [
            issue for issue in self.current_sprint['issues']
            if issue.get('Assignee') == member_name
        ]

    def build_tasks_context(self, member_name: str) -> str:
        """Build context string for member's tasks."""
        tasks = self.get_member_tasks(member_name)
        if not tasks:
            return "No tasks assigned currently."
        return "\n".join([
            f"- {task['Key']}: {task['Summary']} (Status: {task['Status']})"
            for task in tasks
        ])

    def get_mongo_context(self, member_name: str) -> str:
        # Retrieve only the latest 3 summaries for this user from previous standups
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
        """
        Fetch relevant context from Pinecone for a given query string.
        Returns a list of dicts with 'text' and other metadata.
        """
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
        Let the LLM decide the next question based on the full, updated context after each answer.
        No manual marking of answered questions; the LLM should avoid repeats based on context.
        """
        # Define standard Scrum questions
        scrum_questions = [
            "What did you work on since the last standup?",
            "What are you planning to work on today?",
            "Are there any blockers or impediments in your way?",
            "Is there anything else you'd like to share with the team?"
        ]
        # Select the base question for this step, or default to the last question if out of range
        if 0 <= step < len(scrum_questions):
            base_question = scrum_questions[step]
        else:
            base_question = scrum_questions[-1]

        # Build the user's task context (from the current sprint)
        tasks_context = self.build_tasks_context(member_name)
        pinecone_context = self.get_contextual_history(member_name)
        mongo_context = self.get_mongo_context(member_name)

        # Try to get the current task_key and task description for this member
        task_key = None
        task_description = None
        if self.current_sprint:
            member_tasks = self.get_member_tasks(member_name)
            if member_tasks:
                task_key = member_tasks[0].get('Key')
                # Use summary or description for semantic search
                task_description = member_tasks[0].get('Summary') or member_tasks[0].get('Description')

        # Fetch cross-user context for this task (exclude current user)
        cross_user_contexts = self.fetch_cross_user_context(task_key, exclude_user_id=self.user_id, top_k=5) if task_key else []
        cross_user_str = ""
        if cross_user_contexts:
            cross_user_str = "\nContradictory or additional context from other users for this task:\n" + "\n".join([
                f"- {c.get('member_name', 'Unknown')}: {c.get('text', '')}" for c in cross_user_contexts
            ])

        # Fetch semantically similar context from other users' tasks (exclude current user)
        semantic_contexts = []
        semantic_context_str = ""
        if task_description:
            semantic_contexts = self.fetch_semantic_cross_user_context(task_description, exclude_user_id=self.user_id, top_k=5)
        if semantic_contexts:
            semantic_context_str = "\nRelated context from other users working on similar tasks:\n" + "\n".join([
                f"- {c.get('member_name', 'Unknown')}: {c.get('text', '')}" for c in semantic_contexts
            ])

        # Get previous question and answer for this member
        previous_question = None
        previous_answer = None
        for msg in reversed(self.conversation_history):
            if msg["role"] == "assistant" and previous_question is None:
                previous_question = msg["content"]
            elif msg["role"] == "user" and previous_answer is None:
                previous_answer = msg["content"]
            if previous_question and previous_answer:
                break

        prompt = f"""
You are an AI Scrum Master named AgileBot conducting a standup with {member_name} at step {step}.

Here is the standard Scrum question you should ask:
"{base_question}"

Previous question asked:
"{previous_question}"

User's previous answer:
"{previous_answer}"

Tasks context for {member_name}:
{tasks_context}

Recent conversation context from Pinecone:
{pinecone_context}

Historical context from MongoDB:
{mongo_context}

{cross_user_str}

{semantic_context_str}

Using the above information, generate a single, friendly, and concise question that incorporates all relevant details.
- Do NOT repeat questions that have already been answered.
- If the user has already addressed a blocker or task, move on to the next relevant topic.
- If clarification is needed, ask a follow-up, otherwise proceed to the next standup question.
"""

        # Call the Gemini model to generate a refined question
        refined_question = model.generate_content(prompt).text.strip()

        # Fallback if LLM returns something empty (rare edge case)
        if not refined_question:
            return "Thank you, all questions have been answered!"
        return refined_question

    def add_user_response(self, member_name: str, response: str):
        """Process and store the user response along with internal analysis and update context immediately."""
        # Add user message to conversation history
        self.conversation_history.append({
            "role": "user",
            "content": response,
            "timestamp": datetime.now(timezone.utc)
        })

        # Create an analysis prompt for the response
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
        # Append the internal analysis as an assistant message.
        analysis_message = f"[Internal Analysis]\n{analysis_result}"
        self.conversation_history.append({
            "role": "assistant",
            "content": analysis_message,
            "timestamp": datetime.now(timezone.utc)
        })



        # Store this conversation turn in Pinecone for future context
        self.store_context_in_pinecone(member_name, response, analysis_result)

    def generate_ai_response(self) -> str:
        """
        Generate a follow-up question for the current conversation step.
        (By design, we use our fixed question mapping so the assistant does not reveal underlying context.)
        """
        # Streamlit session state removed; this method should not be used in backend-only context
        raise NotImplementedError("generate_ai_response is not supported in backend-only mode.")

    def add_assistant_response(self, response: str):
        """Store the assistant's response in conversation history."""
        self.conversation_history.append({
            "role": "assistant",
            "content": response,
            "timestamp": datetime.now(timezone.utc)
        })

    def check_response_completeness(self, member_name: str, response: str) -> bool:
        """
        Analyze the response to determine if it is complete.
        If the response is trivial (like 'nothing' or 'no'), consider it complete.
        Otherwise, use the LLM to analyze further.
        """
        normalized = response.strip().lower()
        if normalized in ["nothing", "nothing thank you", "no", "none"]:
            return True  # Treat these as complete responses

        prompt = f"""
You are an AI Scrum Master. Analyze the following standup response from {member_name}:
---
{response}
---
Consider the response 'Complete' if the user gives any reasonable update, even if it is brief, informal, or non-technical (e.g., 'I'm working on it', 'All good', 'No blockers', 'Progressing', etc.).
Only mark as 'Incomplete' if the response is completely missing, off-topic, or does not address the question at all.
Examples of responses that should be considered 'Complete':
- 'I'm working on it'
- 'No blockers'
- 'Progressing'
- 'All good'
- 'Done'
- 'Fixed'
- 'Still working'
Answer with a single word: "Complete" if the response is adequate, or "Incomplete" if further follow-up is needed. Then, provide a brief explanation.
"""
        result = model.generate_content(prompt).text.strip()
        print("Completeness Analysis:", result)
        if result.lower().startswith("complete"):
            return True
        return False


    def generate_summary(self) -> str:
        """Generate a summary of the standup."""
        # Only use the last 10 messages for summary to avoid token overflow
        recent_history = self.conversation_history[-10:]
        # Gather all unique team members who participated in this standup
        participants = set()
        for msg in recent_history:
            if isinstance(msg, dict) and msg.get("role") == "user":
                # Try to extract the member name from the message if available
                # (Assumes the member name is in the message or can be tracked elsewhere)
                # If not available, this will just collect all users who responded
                pass  # You can enhance this logic if you track member names per message
        if hasattr(self, "team_members"):
            participants = self.team_members
        summary_prompt = f"""
Summarize the following standup conversation:
---
{recent_history}
---
For each of these team members, provide a summary of their updates (even if brief): {', '.join(participants) if participants else '[team members not found]'}
Include:
- Key updates per team member
- Identified blockers
- Action items/follow-ups
- Overall sprint progress
Format the summary in markdown.
"""
        return model.generate_content(summary_prompt).text.strip()

    # --------------------------------------------------------------------------------
    # Pinecone Context Management Functions
    # --------------------------------------------------------------------------------
    def store_context_in_pinecone(self, member_name: str, response: str, analysis_result: str):
        """
        Store the user's response and analysis in Pinecone, including a task_key for cross-user context.
        """
        if not index:
            return []
        # Try to get the current task key (if any)
        task_key = None
        if self.current_sprint:
            member_tasks = self.get_member_tasks(member_name)
            if member_tasks:
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
            # Remove Streamlit dependency for conversation_step
            "conversation_step": 1,
            "task_key": task_key
        }
        sprint_id = self.current_sprint.get('id') if self.current_sprint else None
        if sprint_id != None:
            metadata['sprint_id'] = sprint_id
        try:
            index.upsert([(vector_id, vector, metadata)])
            self.context_cache.pop(member_name, None)
        except Exception as e:
            print(f"Failed to store context in Pinecone: {str(e)}")

    def fetch_cross_user_context(self, task_key: Optional[str], exclude_user_id: Optional[str] = None, top_k: int = 5) -> list:
        """
        Fetch context for a given task_key from all users except the current one.
        Returns a list of dicts with member_name, user_id, and text.
        Pinecone filter syntax: {"task_key": "...", "user_id": {"$ne": "..."}}
        """
        if not index or not task_key:
            return []
        try:
            filter_dict = {"task_key": {"$eq": task_key}}
            if exclude_user_id:
                filter_dict["user_id"] = {"$ne": exclude_user_id}
            results = index.query(
                vector=np.zeros(EMBEDDING_DIMENSION).tolist(),  # dummy vector, just filter by metadata
                top_k=top_k,
                include_metadata=True,
                filter=filter_dict #type: ignore
            )
            return [match.metadata for match in results.matches]#type: ignore
        except Exception as e:
            print(f"Failed to fetch cross-user context from Pinecone: {str(e)}")
            return []

    def fetch_semantic_cross_user_context(self, task_description: str, exclude_user_id: Optional[str] = None, top_k: int = 5) -> list:
        """
        Fetch context for semantically similar tasks from all users except the current one.
        Returns a list of dicts with member_name, user_id, and text.
        """
        if not index or not task_description:
            return []
        try:
            vector = safe_encode(embedding_model, task_description)
            filter_dict = {}
            if exclude_user_id:
                filter_dict["user_id"] = {"$ne": exclude_user_id}
            # Always pass a dict for filter, even if empty
            results = index.query(
                vector=vector,
                top_k=top_k,
                include_metadata=True,
                filter=filter_dict
            )
            matches = getattr(results, "matches", [])
            # Only include matches above a similarity threshold
            threshold = 0.7  # Cosine similarity threshold
            return [
                match.metadata for match in matches
                if hasattr(match, "metadata") and getattr(match, "score", 0) >= threshold
            ]
        except Exception as e:
            print(f"Failed to fetch semantic cross-user context: {str(e)}")
            return []
