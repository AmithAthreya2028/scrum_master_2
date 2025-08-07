import os
import re
from typing import Optional
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from dotenv import load_dotenv
import uvicorn
import uuid
import requests

# Import core functionality from the original file
from safe import (
    AIScrumMaster,
    get_boards,
    store_conversation
)

# Load environment variables
load_dotenv()

app = FastAPI(title="MS Teams Agentic Scrum")


# Session storage for bot state (in-memory for simplicity)
# In production, use a database or Redis for persistence
bot_sessions = {}

# Hardcoded Teams user IDs for each member (from userid.txt)
TEAM_MEMBER_IDS = {
    "amith": "29:1EaTSFcrZkjsH3S6cgT9h88AzHL4AqFSmyE2I7LfhIGBu3QpI9aBXxuB0ChMkDWSRYYQEQuUK6Lc-mGoNzB58FA",
    "rahuk": "29:1XFcY26AqQUhnEbXtLhwS92rQRZyu_bxLNHQiswlKNKffy9as-xn2rpVmArUCfUtAC8eEaS2O9ZWvM4fOSe4NdA",
    "sachin": "29:1wIHM0iaGI3IKzW-fHfzzPQ2K5E1MbVjRVoNK2OnodxBhotBQmxgSX1fCB-8laPQbCEEhUzw-PPTyVt_YML3HzQ",
    "yuvraj": "29:15PQxlsgRCbsfORrMJT0-tjx2986KF1qJCW1b8YwD62ocx37V5R6zd6gNY7jNacSTdZZRwUTiPq7N10zwfDph7Q"
}


class BotRequest(BaseModel):
    activity_id: str
    activity_type: str
    text: str = ""
    user_id: str
    conversation_id: str
    session_id: Optional[str] = None

class BotResponse(BaseModel):
    activity_id: str
    text: str
    session_id: str
    is_end: bool = False
    summary: Optional[str] = None
    requires_input: bool = True

def get_or_create_session(session_id: Optional[str] = None, user_id: str = "unknown_user"):
    """Get existing session or create a new one"""
    # Remove user_id dependency for session uniqueness
    if session_id and session_id in bot_sessions:
        return session_id, bot_sessions[session_id]
    new_session_id = session_id or str(uuid.uuid4())
    bot_sessions[new_session_id] = {
        "user_id": user_id,
        "scrum_master": None,
        "standup_started": False,
        "current_member_index": 0,
        "conversation_step": 1,
        "messages": [],
    "team_members": [
        {"name": "amith", "teams_id": TEAM_MEMBER_IDS["amith"]},
        {"name": "rahuk", "teams_id": TEAM_MEMBER_IDS["rahuk"]},
        {"name": "sachin", "teams_id": TEAM_MEMBER_IDS["sachin"]},
        {"name": "yuvraj", "teams_id": TEAM_MEMBER_IDS["yuvraj"]}
    ],
        "selected_board_id": None,
        "nothing_count": 0,
        "show_summary": False
    }
    return new_session_id, bot_sessions[new_session_id]

def get_display_name(member_data):
    """Extract a clean display name from member data regardless of format"""
    if isinstance(member_data, str):
        return member_data
    elif isinstance(member_data, tuple) and len(member_data) >= 2:
        # Handle tuple format (id, display_name)
        return member_data[1] if isinstance(member_data[1], str) else "Team Member"
    elif isinstance(member_data, dict):
        # Handle dictionary format
        return member_data.get('displayName', member_data.get('name', 'Team Member')) if isinstance(member_data, dict) else 'Team Member'
    else:
        # Fallback for unknown formats
        return "Team Member"

@app.get("/")
async def root():
    return {"status": "online", "message": "MS Teams Agentic Scrum API"}

@app.get("/debug-gemini-key")
async def debug_gemini_key():
    import os
    print("GEMINI_API_KEY on /debug-gemini-key call:", os.getenv("GEMINI_API_KEY"))
    return {"gemini_api_key": os.getenv("GEMINI_API_KEY")}

@app.get("/boards")
async def list_boards():
    """List available JIRA boards"""
    try:
        boards = get_boards()
        return {"boards": boards}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch boards: {str(e)}")

@app.post("/start", response_model=BotResponse)
async def start_session(request: BotRequest):
    """Start a new bot session"""
    safe_session_id = request.session_id if request.session_id is not None else str(uuid.uuid4())
    safe_user_id = request.user_id if request.user_id is not None else "unknown_user"
    session_id, session = get_or_create_session(safe_session_id, safe_user_id)

    # Try to import get_last_selected_board from ms_teams_agentic_scrum
    try:
        from ms_teams_agentic_scrum import get_last_selected_board
        last_board_id = get_last_selected_board(safe_user_id)
    except Exception:
        last_board_id = None

    if last_board_id:
        session["selected_board_id"] = last_board_id
        # Initialize scrum master and start standup as if board was just selected
        session["scrum_master"] = AIScrumMaster(safe_user_id)
        if session["scrum_master"].initialize_sprint_data(last_board_id):
            # Store team members with proper display names
            session["team_members"] = list(session["scrum_master"].team_members)
            # Patch: Ensure each member dict has the correct Teams user ID
            for member in session["team_members"]:
                # Try to match by lowercased name (as in TEAM_MEMBER_IDS)
                name_key = None
                if isinstance(member, dict):
                    name_key = member.get("name") or member.get("displayName")
                elif isinstance(member, str):
                    name_key = member
                if name_key:
                    key_lc = name_key.strip().lower()
                    for k in TEAM_MEMBER_IDS:
                        if k.lower() == key_lc:
                            member["teams_id"] = TEAM_MEMBER_IDS[k]
                            break
            if not session["team_members"]:
                return BotResponse(
                    activity_id=request.activity_id,
                    text="No team members found in the active sprint for this board. Please ensure issues are assigned in JIRA.",
                    session_id=session_id,
                    requires_input=True
                )
            session["standup_started"] = True
            session["current_member_index"] = 0
            session["conversation_step"] = 1
            session["messages"] = []
            session["nothing_count"] = 0
            session["show_summary"] = False

            # Get first team member and ensure we have a proper display name
            member_data = session["team_members"][0]
            member_display_name = get_display_name(member_data)
            
            question = session["scrum_master"].generate_question(
                member_data,
                session["conversation_step"]
            )
            session["messages"].append({
                "role": "assistant",
                "content": question
            })
            session["scrum_master"].add_assistant_response(question, member_data)
            return BotResponse(
                activity_id=request.activity_id,
                text=f"Welcome back! Using your last selected board (ID: {last_board_id}).\nStarting standup with {member_display_name}.\n\n{question}",
                session_id=session_id,
                requires_input=True
            )
        else:
            boards = get_boards()
            board_text = "Please select a board by sending its ID:\n"
            for board in boards:
                board_text += f"- {board.get('name', 'Unknown')} (ID: {board.get('id', 'N/A')})\n"
            return BotResponse(
                activity_id=request.activity_id,
                text="Failed to initialize sprint data. Please try another board.\n\n" + board_text,
                session_id=session_id,
                requires_input=True
            )

    # If no last board, prompt for selection as before
    boards = get_boards()
    if not boards:
        return BotResponse(
            activity_id=request.activity_id,
            text="No boards available. Please check your JIRA configuration.",
            session_id=session_id,
            requires_input=False
        )
    board_text = "Please select a board by sending its ID:\n"
    for board in boards:
        board_text += f"- {board.get('name', 'Unknown') if isinstance(board, dict) else 'Unknown'} (ID: {board.get('id', 'N/A') if isinstance(board, dict) else 'N/A'})\n"

    return BotResponse(
        activity_id=request.activity_id,
        text="Welcome to AI Scrum Master! Let me fetch the available boards.\n\n" + board_text,
        session_id=session_id,
        requires_input=True
    )

@app.post("/select_board", response_model=BotResponse)
async def select_board(request: BotRequest):
    """Select a board, initialize team members, and immediately start the standup"""
    # Ensure session_id is a string (not None)
    safe_session_id = request.session_id if request.session_id is not None else str(uuid.uuid4())
    safe_user_id = request.user_id if request.user_id is not None else "unknown_user"
    session_id, session = get_or_create_session(safe_session_id, safe_user_id)

    import re
    # Remove Teams mention markup and bot name from the message
    clean_text = re.sub(r"<at>.*?</at>", "", request.text).replace("@Agentic Scrum Bot", "").strip()
    try:
        board_id = int(clean_text)
    except ValueError:
        boards = get_boards()
        board_text = "Please select a board by sending its ID:\n"
        for board in boards:
            board_text += f"- {board.get('name', 'Unknown') if isinstance(board, dict) else 'Unknown'} (ID: {board.get('id', 'N/A') if isinstance(board, dict) else 'N/A'})\n"
        return BotResponse(
            activity_id=request.activity_id,
            text="Invalid board ID. Please send a numeric ID.\n\n" + board_text,
            session_id=session_id,
            requires_input=True
        )

    # Fetch available boards and check validity
    boards = get_boards()
    valid_board_ids = {board.get('id') if isinstance(board, dict) else None for board in boards}
    if board_id not in valid_board_ids:
        board_text = "Please select a board by sending its ID:\n"
        for board in boards:
            board_text += f"- {board.get('name', 'Unknown') if isinstance(board, dict) else 'Unknown'} (ID: {board.get('id', 'N/A') if isinstance(board, dict) else 'N/A'})\n"
        return BotResponse(
            activity_id=request.activity_id,
            text=f"Invalid board ID. Please select from the following boards:\n\n{board_text}",
            session_id=session_id,
            requires_input=True
        )

    # Save selected board
    session["selected_board_id"] = board_id

    # Store last selected board in MongoDB
    from ms_teams_agentic_scrum import set_last_selected_board
    set_last_selected_board(safe_user_id, board_id)

    # Initialize scrum master and get team members
    session["scrum_master"] = AIScrumMaster(safe_user_id)
    if session["scrum_master"].initialize_sprint_data(board_id):
        # Store team members with proper handling for display names
        session["team_members"] = list(session["scrum_master"].team_members)
        # Patch: Ensure each member dict has the correct Teams user ID
        for member in session["team_members"]:
            name_key = None
            if isinstance(member, dict):
                name_key = member.get("name") or member.get("displayName")
            elif isinstance(member, str):
                name_key = member
            if name_key:
                key_lc = name_key.strip().lower()
                for k in TEAM_MEMBER_IDS:
                    if k.lower() == key_lc:
                        member["teams_id"] = TEAM_MEMBER_IDS[k]
                        break
        if not session["team_members"]:
            return BotResponse(
                activity_id=request.activity_id,
                text="No team members found in the active sprint for this board. Please ensure issues are assigned in JIRA.",
                session_id=session_id,
                requires_input=True
            )

        # Start the standup with the first team member
        session["standup_started"] = True
        session["current_member_index"] = 0
        session["conversation_step"] = 1
        session["messages"] = []
        session["nothing_count"] = 0
        session["show_summary"] = False

        # Get first team member and ensure we have a proper display name
        member_data = session["team_members"][0]
        member_display_name = get_display_name(member_data)
        
        question = session["scrum_master"].generate_question(
            member_data,
            session["conversation_step"]
        )
        session["messages"].append({
            "role": "assistant",
            "content": question
        })
        session["scrum_master"].add_assistant_response(question, member_data)
        return BotResponse(
            activity_id=request.activity_id,
            text=f"Starting standup with {member_display_name}.\n\n{question}",
            session_id=session_id,
            requires_input=True
        )
    else:
        boards = get_boards()
        board_text = "Please select a board by sending its ID:\n"
        for board in boards:
            board_text += f"- {board.get('name', 'Unknown') if isinstance(board, dict) else 'Unknown'} (ID: {board.get('id', 'N/A') if isinstance(board, dict) else 'N/A'})\n"
        return BotResponse(
            activity_id=request.activity_id,
            text="Failed to initialize sprint data. Please try another board.\n\n" + board_text,
            session_id=session_id,
            requires_input=True
        )

# Removed /select_user endpoint and logic, as member selection is no longer needed.

@app.post("/message", response_model=BotResponse)
async def process_message(request: BotRequest):
    """Process a message in an ongoing standup"""
    safe_session_id = request.session_id if request.session_id is not None else str(uuid.uuid4())
    safe_user_id = request.user_id if request.user_id is not None else "unknown_user"
    session_id, session = get_or_create_session(safe_session_id, safe_user_id)

    if not session["standup_started"]:
        import re
        clean_response = re.sub(r"<at>.*?</at>", "", request.text).replace("@Agentic Scrum Bot", "").strip()
        if clean_response.lower() in ["stop", "cancel", "abort", "quit"]:
            return BotResponse(
                activity_id=request.activity_id,
                text="No standup is currently in progress. Type 'start' to begin a new standup session.",
                session_id=session_id,
                requires_input=True
            )
        if clean_response.lower() in ["start", "start standup"]:
            session["selected_board_id"] = None
            boards = get_boards()
            if not boards:
                return BotResponse(
                    activity_id=request.activity_id,
                    text="No boards available. Please check your JIRA configuration.",
                    session_id=session_id,
                    requires_input=False
                )
            board_text = "Please select a board by sending its ID:\n"
            for board in boards:
                board_text += f"- {board.get('name', 'Unknown')} (ID: {board.get('id', 'N/A')})\n"
            return BotResponse(
                activity_id=request.activity_id,
                text="Welcome to AI Scrum Master! Let me fetch the available boards.\n\n" + board_text,
                session_id=session_id,
                requires_input=True
            )
        else:
            return BotResponse(
                activity_id=request.activity_id,
                text="Standup not started. Please type 'start' to begin a new standup session.",
                session_id=session_id,
                requires_input=True
            )

    team_members = session["team_members"]
    current_index = session["current_member_index"]
    show_summary = session.get("show_summary", False)

    if show_summary or current_index >= len(team_members):
        summary = session["scrum_master"].generate_summary()
        conversation_doc = {
            "user_id": session["user_id"],
            "messages": session["scrum_master"].conversation_history,
            "summary": summary
        }
        store_conversation(conversation_doc)
        session["standup_started"] = False
        session["current_member_index"] = 0
        session["conversation_step"] = 1
        session["messages"] = []
        session["show_summary"] = False
        return BotResponse(
            activity_id=request.activity_id,
            text="Standup Summary:\n\n" + summary + "\n\nIf you'd like to start another standup, type 'start'.",
            session_id=session_id,
            is_end=True,
            summary=summary,
            requires_input=True
        )

    # Get current member data and ensure we have a proper display name
    member_data = team_members[current_index]
    member_display_name = get_display_name(member_data)
    # Try to extract both JIRA and Teams names if available
    jira_name = None
    ms_teams_name = None
    if isinstance(member_data, dict):
        jira_name = member_data.get('displayName') or member_data.get('name') if isinstance(member_data, dict) else None
        ms_teams_name = member_data.get('ms_teams_name') or member_data.get('teamsName') if isinstance(member_data, dict) else None
    elif isinstance(member_data, str):
        jira_name = member_data
        ms_teams_name = member_data
    elif isinstance(member_data, tuple):
        # (id, display_name)
        jira_name = member_data[1] if len(member_data) > 1 else member_data[0]
        ms_teams_name = member_data[1] if len(member_data) > 1 else member_data[0]

    response = request.text
    import re
    clean_response = re.sub(r"<at>.*?</at>", "", response).replace("@Agentic Scrum Bot", "").strip()

    # --- Per-person auth check using Teams user ID ---
    if not is_authorized_user(session, request):
        return BotResponse(
            activity_id=request.activity_id,
            text=f"Waiting for a response from {member_display_name}. Only {member_display_name} can answer this question.",
            session_id=session_id,
            requires_input=True
        )
    # --- End user check ---

    if clean_response.lower() in ["change board", "switch board"]:
        session["selected_board_id"] = None
        boards = get_boards()
        board_text = "Please select a board by sending its ID:\n"
        for board in boards:
            board_text += f"- {board.get('name', 'Unknown') if isinstance(board, dict) else 'Unknown'} (ID: {board.get('id', 'N/A') if isinstance(board, dict) else 'N/A'})\n"
        return BotResponse(
            activity_id=request.activity_id,
            text="Board selection reset. Please select a new board:\n\n" + board_text,
            session_id=session_id,
            requires_input=True
        )

    if clean_response.lower() in ["end standup", "end", "finish"]:
        summary = session["scrum_master"].generate_summary()
        conversation_doc = {
            "user_id": session["user_id"],
            "messages": session["scrum_master"].conversation_history,
            "summary": summary
        }
        store_conversation(conversation_doc)
        session["standup_started"] = False
        session["current_member_index"] = 0
        session["conversation_step"] = 1
        session["messages"] = []
        session["show_summary"] = False
        return BotResponse(
            activity_id=request.activity_id,
            text="Standup Summary:\n\n" + summary + "\n\nIf you'd like to start another standup, type 'start'.",
            session_id=session_id,
            is_end=True,
            summary=summary,
            requires_input=True
        )

    if clean_response.lower() in ["stop", "cancel", "abort", "quit"]:
        session["standup_started"] = False
        session["show_summary"] = False
        session["current_member_index"] = 0
        session["conversation_step"] = 1
        session["messages"] = []
        session["nothing_count"] = 0
        session["selected_board_id"] = None
        session["team_members"] = []
        session["scrum_master"] = None
        partial_summary = session["scrum_master"].generate_summary() if session.get("scrum_master") else "Standup stopped. No summary available."
        conversation_doc = {
            "user_id": session["user_id"],
            "messages": session["scrum_master"].conversation_history if session.get("scrum_master") else [],
            "summary": partial_summary
        }
        store_conversation(conversation_doc)
        return BotResponse(
            activity_id=request.activity_id,
            text="Standup has been stopped by request. Here’s a summary of what was discussed so far:\n\n" + partial_summary + "\n\nYou can type 'start' to begin a new standup session.",
            session_id=session_id,
            is_end=True,
            summary=partial_summary,
            requires_input=True
        )

    if clean_response.lower() in ["skip", "skip user", "not available", "on leave", "sick leave"]:
        skipped_member_display_name = member_display_name
        session["messages"].append({
            "role": "system",
            "content": f"{skipped_member_display_name} was skipped (not available)."
        })
        if session.get("scrum_master"):
            session["scrum_master"].add_assistant_response(f"{skipped_member_display_name} was skipped (not available).", member_data)
        session["current_member_index"] += 1
        session["conversation_step"] = 1
        session["messages"] = []
        session["nothing_count"] = 0
        if session["current_member_index"] >= len(team_members):
            session["show_summary"] = True
            summary = session["scrum_master"].generate_summary()
            conversation_doc = {
                "user_id": session["user_id"],
                "messages": session["scrum_master"].conversation_history,
                "summary": summary
            }
            store_conversation(conversation_doc)
            session["standup_started"] = False
            session["current_member_index"] = 0
            session["conversation_step"] = 1
            session["messages"] = []
            session["show_summary"] = False
            return BotResponse(
                activity_id=request.activity_id,
                text="Standup Summary:\n\n" + summary + "\n\nIf you'd like to start another standup, type 'start'.",
                session_id=session_id,
                is_end=True,
                summary=summary,
                requires_input=True
            )
        next_member_data = team_members[session["current_member_index"]]
        next_member_display_name = get_display_name(next_member_data)
        next_question = session["scrum_master"].generate_question(
            next_member_data,
            session["conversation_step"]
        )
        session["messages"].append({
            "role": "assistant",
            "content": next_question
        })
        session["scrum_master"].add_assistant_response(next_question, next_member_data)
        return BotResponse(
            activity_id=request.activity_id,
            text=f"{skipped_member_display_name} was skipped. Moving on to {next_member_display_name}.\n\n{next_question}",
            session_id=session_id,
            requires_input=True
        )

    session["messages"].append({
        "role": "user",
        "content": clean_response
    })
    session["scrum_master"].add_user_response(member_data, clean_response)

    trivial_responses = [
        "nothing", "nothing thank you", "no", "none", "done", "finished", "ok",
        "nope", "no nothing", "nothing to discuss", "no nothing to discuss further"
    ]
    if clean_response.strip().lower() in trivial_responses:
        session["nothing_count"] = session.get("nothing_count", 0) + 1
    else:
        session["nothing_count"] = 0

    num_questions = 4

    if session["conversation_step"] >= num_questions or session["nothing_count"] >= 2:
        final_message = f"Thanks for the update, {member_display_name}."
        session["current_member_index"] += 1
        session["conversation_step"] = 1
        session["messages"] = []
        session["nothing_count"] = 0
        if session["current_member_index"] >= len(team_members):
            session["show_summary"] = True
            summary = session["scrum_master"].generate_summary()
            conversation_doc = {
                "user_id": session["user_id"],
                "messages": session["scrum_master"].conversation_history,
                "summary": summary
            }
            store_conversation(conversation_doc)
            session["standup_started"] = False
            session["current_member_index"] = 0
            session["conversation_step"] = 1
            session["messages"] = []
            session["show_summary"] = False
            return BotResponse(
                activity_id=request.activity_id,
                text="Standup Summary:\n\n" + summary + "\n\nIf you'd like to start another standup, type 'start'.",
                session_id=session_id,
                is_end=True,
                summary=summary,
                requires_input=True
            )
        next_member_data = team_members[session["current_member_index"]]
        next_member_display_name = get_display_name(next_member_data)
        next_question = session["scrum_master"].generate_question(
            next_member_data,
            session["conversation_step"]
        )
        session["messages"].append({
            "role": "assistant",
            "content": next_question
        })
        session["scrum_master"].add_assistant_response(next_question, next_member_data)
        return BotResponse(
            activity_id=request.activity_id,
            text=f"{final_message} I'll now move on to {next_member_display_name}.\n\n{next_question}",
            session_id=session_id,
            requires_input=True
        )
    else:
        session["conversation_step"] += 1
        next_question = session["scrum_master"].generate_question(
            member_data,
            session["conversation_step"]
        )
        session["messages"].append({
            "role": "assistant",
            "content": next_question
        })
        session["scrum_master"].add_assistant_response(next_question, member_data)
        return BotResponse(
            activity_id=request.activity_id,
            text=next_question,
            session_id=session_id,
            requires_input=True
        )

MICROSOFT_APP_ID = os.getenv("MICROSOFT_APP_ID")
MICROSOFT_APP_PASSWORD = os.getenv("MICROSOFT_APP_PASSWORD")

def send_teams_reply(service_url, conversation_id, reply_to_id, text, app_id, app_password):
    # Get OAuth token
    token_url = "https://login.microsoftonline.com/botframework.com/oauth2/v2.0/token"
    data = {
        "grant_type": "client_credentials",
        "client_id": app_id,
        "client_secret": app_password,
        "scope": "https://api.botframework.com/.default"
    }
    token_response = requests.post(token_url, data=data)
    token = token_response.json().get("access_token")

    # Prepare the reply activity
    activity = {
        "type": "message",
        "text": text
    }

    # Send the reply
    url = f"{service_url}v3/conversations/{conversation_id}/activities/{reply_to_id}"
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }
    response = requests.post(url, headers=headers, json=activity)
    print("Teams proactive reply status:", response.status_code, response.text)
    return response

@app.post("/webhook")
async def teams_webhook(request: Request):
    """Webhook endpoint for MS Teams Bot Framework"""
    try:
        data = await request.json()
        print("=== /webhook called ===")
        print("Raw payload:", data)
        print("Received activity type:", data.get("type"))

        # Extract relevant information from Teams activity
        activity_id = data.get("iit d", str(uuid.uuid4()))
        activity_type = data.get("type", "")
        text = data.get("text", "").strip()

        # Extract user and conversation identifiers
        from_obj = data.get("from", {})
        user_id = from_obj.get("id", "unknown")
        # user_name = from_obj.get("name", "unknown")

        conversation = data.get("conversation", {})
        conversation_id = conversation.get("id", "unknown")

        # Get session ID from channelData if available
        channel_data = data.get("channelData", {})
        session_id = channel_data.get("sessionId", None)

        # Fallback to conversation_id if session_id is missing
        if not session_id:
            session_id = conversation_id

        bot_response = None

        # Check if this is a new conversation
        if activity_type == "conversationUpdate":
            bot_request = BotRequest(
                activity_id=activity_id,
                activity_type=activity_type,
                text="",
                user_id=user_id,
                conversation_id=conversation_id,
                session_id=session_id
            )
            try:
                bot_response = await start_session(bot_request)
                print("start_session response:", bot_response)
            except Exception as e:
                print("Exception in start_session:", e)
                return JSONResponse(content={"message": "Internal error in start_session"})

        # Handle message activity
        elif activity_type == "message":
            print("Handling message activity")
            try:
                # Check if we need to start or we're in an existing session
                bot_request = BotRequest(
                    activity_id=activity_id,
                    activity_type=activity_type,
                    text=text,
                    user_id=user_id,
                    conversation_id=conversation_id,
                    session_id=session_id
                )

                # Determine the action based on session state
                if not session_id or session_id not in bot_sessions:
                    print("No session found, starting new session")
                    bot_response = await start_session(bot_request)
                    print("start_session response:", bot_response)
                else:
                    session = bot_sessions[session_id]
                    if session.get("selected_board_id") is None:
                        print("Board selection phase")
                        # Try to parse the board ID from the message
                        import re
                        clean_text = re.sub(r"<at>.*?</at>", "", text).replace("@Agentic Scrum Bot", "").strip()
                        try:
                            board_id = int(clean_text)
                            bot_response = await select_board(bot_request)
                            print("select_board response:", bot_response)
                        except ValueError:
                            # Not a valid board ID, prompt user to send a valid one
                            boards = get_boards()
                            board_text = "Please select a board by sending its ID:\n"
                            for board in boards:
                                board_text += f"- {board.get('name', 'Unknown')} (ID: {board.get('id', 'N/A')})\n"
                            bot_response = BotResponse(
                                activity_id=activity_id,
                                text="Invalid board ID. Please send a numeric ID.\n\n" + board_text,
                                session_id=session_id,
                                requires_input=True
                            )
                    elif not session.get("standup_started"):
                        print("Standup not started, waiting for board selection.")
                        # This should not happen, but if it does, prompt to select a board again
                        bot_response = await start_session(bot_request)
                        print("start_session response:", bot_response)
                    else:
                        print("Processing message in ongoing standup")
                        bot_response = await process_message(bot_request)
                        print("process_message response:", bot_response)
            except Exception as e:
                print("Exception in message handling:", e)
                return JSONResponse(content={"message": "Internal error in message handling"})
        else:
            print("Unsupported activity type received:", activity_type)
            return JSONResponse(content={"message": f"Activity type '{activity_type}' not supported"})

        # Send reply for personal chat or group chat
        if bot_response is not None:
            conversation_type = data.get("conversation", {}).get("conversationType")
            if conversation_type == "personal":
                print("Sending proactive reply for personal chat.")
                send_teams_reply(
                    service_url=data.get("serviceUrl"),
                    conversation_id=data.get("conversation", {}).get("id"),
                    reply_to_id=data.get("id"),
                    text=bot_response.text,
                    app_id=MICROSOFT_APP_ID,
                    app_password=MICROSOFT_APP_PASSWORD
                )
                # Return 200 OK with empty body (Teams will show the proactive reply)
                return {}
            elif conversation_type in ["groupChat", "channel"]:
                print("Sending reply for group chat or channel using Teams REST API.")
                # Send reply using Teams REST API for group chat/channel
                service_url = data.get("serviceUrl")
                conversation_id = data.get("conversation", {}).get("id")
                text = bot_response.text
                app_id = MICROSOFT_APP_ID
                app_password = MICROSOFT_APP_PASSWORD

                # Get OAuth token
                token_url = "https://login.microsoftonline.com/botframework.com/oauth2/v2.0/token"
                token_data = {
                    "grant_type": "client_credentials",
                    "client_id": app_id,
                    "client_secret": app_password,
                    "scope": "https://api.botframework.com/.default"
                }
                token_response = requests.post(token_url, data=token_data)
                token = token_response.json().get("access_token")

                # Prepare the reply activity
                activity = {
                    "type": "message",
                    "text": text
                }

                # Send the reply to the conversation activities endpoint
                url = f"{service_url}/v3/conversations/{conversation_id}/activities"
                headers = {
                    "Authorization": f"Bearer {token}",
                    "Content-Type": "application/json"
                }
                response = requests.post(url, headers=headers, json=activity)
                print("Teams group reply status:", response.status_code, response.text)

                # If this is the summary, reset the session only after sending the summary
                if bot_response.is_end:
                    print("Standup complete, summary sent to group chat.")

                # Return 200 OK with empty body
                return {}
            else:
                print("Returning Teams-compatible activity:", bot_response.text)
                return {
                    "type": "message",
                    "text": bot_response.text,
                    "replyToId": data.get("id")
                }
        else:
            print("No response generated.")
            return JSONResponse(content={"message": "No response generated."})

    except Exception as e:
        print("Exception in /webhook endpoint:", e)
        return JSONResponse(
            status_code=500,
            content={"error": f"Failed to process request: {str(e)}"}
        )

def normalize_name(name: str) -> str:
    """Normalize a name for robust comparison."""
    import re
    if not name:
        return ""

    # Lowercase, strip, collapse spaces, replace underscores/dots/dashes with space
    name = name.lower().strip()
    name = re.sub(r"[_\-.]", " ", name)
    name = re.sub(r"\s+", " ", name)
    return name


# Update per-person auth check in process_message (or wherever you check who can answer)
def is_authorized_user(session, request):
    """Check if the incoming user is the expected one for this turn."""
    team_members = session.get("team_members", [])
    current_index = session.get("current_member_index", 0)
    if not team_members or current_index >= len(team_members):
        return True  # fallback: allow
    expected_member = team_members[current_index]
    expected_teams_id = expected_member.get("teams_id") if isinstance(expected_member, dict) else None
    return request.user_id == expected_teams_id

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run("fastapi_teams_bot:app", host="0.0.0.0", port=port)