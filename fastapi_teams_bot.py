import os
from typing import Optional
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from dotenv import load_dotenv
import uvicorn
import uuid
import requests

# Import core functionality from the original file
from ms_teams_agentic_scrum import (
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
    if session_id and session_id in bot_sessions:
        return session_id, bot_sessions[session_id]

    # Create new session
    new_session_id = session_id or str(uuid.uuid4())
    bot_sessions[new_session_id] = {
        "user_id": user_id,
        "scrum_master": None,
        "standup_started": False,
        "current_member_index": 0,
        "conversation_step": 1,
        "messages": [],
        "team_members": [],
        "selected_board_id": None,
        "nothing_count": 0,
        "show_summary": False
    }
    return new_session_id, bot_sessions[new_session_id]

@app.get("/")
async def root():
    return {"status": "online", "message": "MS Teams Agentic Scrum API"}

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
    # Ensure session_id is a string (not None)
    safe_session_id = request.session_id if request.session_id is not None else str(uuid.uuid4())
    safe_user_id = request.user_id if request.user_id is not None else "unknown_user"
    session_id, session = get_or_create_session(safe_session_id, safe_user_id)

    # Initialize response
    response = BotResponse(
        activity_id=request.activity_id,
        text="Welcome to AI Scrum Master! Let me fetch the available boards.",
        session_id=session_id,
        requires_input=False
    )

    # Fetch boards
    boards = get_boards()

    if not boards:
        response.text = "No boards available. Please check your JIRA configuration."
        return response

    # Format boards as text
    board_text = "Please select a board by sending its ID:\n"
    for board in boards:
        board_text += f"- {board.get('name', 'Unknown')} (ID: {board.get('id', 'N/A')})\n"

    response.text = board_text
    response.requires_input = True

    return response

@app.post("/select_board", response_model=BotResponse)
async def select_board(request: BotRequest):
    """Select a board and initialize team members"""
    # Ensure session_id is a string (not None)
    safe_session_id = request.session_id if request.session_id is not None else str(uuid.uuid4())
    safe_user_id = request.user_id if request.user_id is not None else "unknown_user"
    session_id, session = get_or_create_session(safe_session_id, safe_user_id)

    try:
        board_id = int(request.text.strip())
    except ValueError:
        return BotResponse(
            activity_id=request.activity_id,
            text="Invalid board ID. Please send a numeric ID.",
            session_id=session_id,
            requires_input=True
        )

    # Save selected board
    session["selected_board_id"] = board_id

    # Initialize temp scrum master to get team members
    temp_scrum_master = AIScrumMaster(request.user_id)
    if temp_scrum_master.initialize_sprint_data(board_id):
        session["team_members"] = list(temp_scrum_master.team_members)

        if not session["team_members"]:
            return BotResponse(
                activity_id=request.activity_id,
                text="No team members found in the active sprint for this board. Please ensure issues are assigned in JIRA.",
                session_id=session_id,
                requires_input=True
            )

        # Format team members as text
        team_members_text = "Please select your JIRA user by sending the corresponding number:\n"
        for i, member in enumerate(session["team_members"], 1):
            team_members_text += f"{i}. {member}\n"

        return BotResponse(
            activity_id=request.activity_id,
            text=team_members_text,
            session_id=session_id,
            requires_input=True
        )
    else:
        return BotResponse(
            activity_id=request.activity_id,
            text="Failed to initialize sprint data. Please try another board.",
            session_id=session_id,
            requires_input=True
        )

@app.post("/select_user", response_model=BotResponse)
async def select_user(request: BotRequest):
    """Select a user and start the standup"""
    safe_session_id = request.session_id if request.session_id is not None else str(uuid.uuid4())
    safe_user_id = request.user_id if request.user_id is not None else "unknown_user"
    session_id, session = get_or_create_session(safe_session_id, safe_user_id)

    if not session["team_members"]:
        return BotResponse(
            activity_id=request.activity_id,
            text="No team members available. Please restart the session.",
            session_id=session_id,
            requires_input=False
        )

    try:
        # Handle both index selection and direct name input
        user_input = request.text.strip()
        selected_user = None

        try:
            # Try to parse as index
            index = int(user_input) - 1
            if 0 <= index < len(session["team_members"]):
                selected_user = session["team_members"][index]
        except ValueError:
            # Try as direct name input
            if user_input in session["team_members"]:
                selected_user = user_input

        if not selected_user:
            return BotResponse(
                activity_id=request.activity_id,
                text="Invalid selection. Please enter a valid number or exact name.",
                session_id=session_id,
                requires_input=True
            )

        # Initialize the scrum master with the selected user
        session["scrum_master"] = AIScrumMaster(selected_user)
        session["scrum_master"].team_members = set(session["team_members"])

        # Ensure current_sprint is set
        board_id = session["selected_board_id"]
        if board_id:
            session["scrum_master"].initialize_sprint_data(board_id)

        # Start the standup
        session["standup_started"] = True
        session["current_member_index"] = 0
        session["conversation_step"] = 1
        session["messages"] = []

        # Generate first question for the first team member
        member = session["team_members"][0]
        question = session["scrum_master"].generate_question(
            member,
            session["conversation_step"]
        )

        session["messages"].append({
            "role": "assistant",
            "content": question
        })

        return BotResponse(
            activity_id=request.activity_id,
            text=f"Starting standup with {member}.\n\n{question}",
            session_id=session_id,
            requires_input=True
        )

    except Exception as e:
        return BotResponse(
            activity_id=request.activity_id,
            text=f"Error starting standup: {str(e)}",
            session_id=session_id,
            requires_input=False
        )

@app.post("/message", response_model=BotResponse)
async def process_message(request: BotRequest):
    """Process a message in an ongoing standup"""
    safe_session_id = request.session_id if request.session_id is not None else str(uuid.uuid4())
    safe_user_id = request.user_id if request.user_id is not None else "unknown_user"
    session_id, session = get_or_create_session(safe_session_id, safe_user_id)

    if not session["standup_started"]:
        return BotResponse(
            activity_id=request.activity_id,
            text="Standup not started. Please start a new session.",
            session_id=session_id,
            requires_input=False
        )

    team_members = session["team_members"]
    current_index = session["current_member_index"]
    show_summary = session.get("show_summary", False)

    # Check if we need to generate a summary
    if show_summary or current_index >= len(team_members):
        summary = session["scrum_master"].generate_summary()

        # Store the conversation in the database
        conversation_doc = {
            "user_id": session["user_id"],
            "messages": session["scrum_master"].conversation_history,
            "summary": summary
        }
        store_conversation(conversation_doc)

        # Reset the session
        session["standup_started"] = False
        session["current_member_index"] = 0
        session["conversation_step"] = 1
        session["messages"] = []
        session["show_summary"] = False

        return BotResponse(
            activity_id=request.activity_id,
            text="Standup Summary:\n\n" + summary,
            session_id=session_id,
            is_end=True,
            summary=summary,
            requires_input=False
        )

    # Handle response for the current team member
    member = team_members[current_index]
    response = request.text

    # Command to end the standup early

    if response.lower() in ["end standup", "end", "finish"]:
        session["show_summary"] = True
        return await process_message(request)  # Recursively call to generate summary

    # Add user response to messages and scrum master
    session["messages"].append({
        "role": "user",
        "content": response
    })
    session["scrum_master"].add_user_response(member, response)

    # Check if the response is trivial
    if response.strip().lower() in ["nothing", "nothing thank you", "no", "none"]:
        session["nothing_count"] = session.get("nothing_count", 0) + 1
    else:
        session["nothing_count"] = 0  # Reset if the response is meaningful

    # Check response completeness
    is_complete = session["scrum_master"].check_response_completeness(member, response)

    # If the response is complete or too many "nothing" responses
    if is_complete or session["nothing_count"] >= 2:
        final_message = f"Thanks for the update, {member}."

        # Move to next team member
        session["current_member_index"] += 1
        session["conversation_step"] = 1
        session["messages"] = []  # Reset messages for the next member
        session["nothing_count"] = 0  # Reset the counter

        # Check if we're done with all team members
        if session["current_member_index"] >= len(team_members):
            # We're done, get the summary
            session["show_summary"] = True
            return await process_message(request)

        # Get the next member
        next_member = team_members[session["current_member_index"]]
        next_question = session["scrum_master"].generate_question(
            next_member,
            session["conversation_step"]
        )

        session["messages"].append({
            "role": "assistant",
            "content": next_question
        })
        session["scrum_master"].add_assistant_response(next_question)

        return BotResponse(
            activity_id=request.activity_id,
            text=f"{final_message} I'll now move on to {next_member}.\n\n{next_question}",
            session_id=session_id,
            requires_input=True
        )
    else:
        # Continue with the same member
        session["conversation_step"] += 1
        next_question = session["scrum_master"].generate_question(
            member,
            session["conversation_step"]
        )

        session["messages"].append({
            "role": "assistant",
            "content": next_question
        })
        session["scrum_master"].add_assistant_response(next_question)

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
        activity_id = data.get("id", str(uuid.uuid4()))
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
                        bot_response = await select_board(bot_request)
                        print("select_board response:", bot_response)
                    elif not session.get("standup_started"):
                        print("User selection phase")
                        bot_response = await select_user(bot_request)
                        print("select_user response:", bot_response)
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

        # Send proactive reply for personal chat
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
            else:
                print("Returning Teams-compatible activity:", bot_response.text)
                return {
                    "type": "message",
                    "text": bot_response.text,
                    "from": data.get("recipient"),
                    "recipient": data.get("from"),
                    "conversation": data.get("conversation"),
                    "replyToId": data.get("id"),
                    "serviceUrl": data.get("serviceUrl")
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

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("fastapi_teams_bot:app", host="0.0.0.0", port=port)
