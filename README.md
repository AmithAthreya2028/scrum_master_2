# MS Teams Agentic Scrum Master

A FastAPI-based application that provides an AI-powered Scrum Master bot for Microsoft Teams, integrating with JIRA, MongoDB, and Pinecone for intelligent standup facilitation.

## Project Structure

```
scrum_master_2/
├── src/
│   ├── __init__.py
│   ├── bot/
│   │   ├── __init__.py
│   │   └── scrum_master.py  # Core bot logic with AIScrumMaster class
│   └── api/
│       ├── __init__.py
│       └── app.py           # FastAPI application
├── tests/
│   ├── agentic_bot_test.py
│   └── check_api.py
├── data/
│   ├── todo.txt
│   └── userid.txt
├── main.py                  # Entry point to run the API
├── requirements.txt         # Python dependencies
└── README.md
```

## Main Components

- **Bot Logic** (`src/bot/scrum_master.py`): Contains the `AIScrumMaster` class that handles conversation flow, JIRA integration, MongoDB storage, and AI-powered question generation using Gemini.

- **API** (`src/api/app.py`): FastAPI application providing REST endpoints for MS Teams integration, session management, and bot interactions.

## Installation

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Set up environment variables (create a `.env` file):
   ```
   GEMINI_API_KEY=your_gemini_api_key
   MONGO_URI=your_mongodb_uri
   JIRA_URL=your_jira_url
   JIRA_EMAIL=your_jira_email
   JIRA_API_TOKEN=your_jira_api_token
   PINECONE_API_KEY=your_pinecone_api_key
   PINECONE_ENVIRONMENT=your_pinecone_region
   PINECONE_INDEX_NAME=your_index_name
   ```

## Running the Application

Run the API server:
```bash
python main.py
```

Or directly:
```bash
python -m src.api.app
```

The API will be available at `http://localhost:8000`.

## Endpoints

- `GET /`: Health check
- `GET /boards`: List JIRA boards
- `POST /start`: Start a new bot session
- `POST /message`: Send a message to the bot
- `POST /select-board`: Select a JIRA board for the session

## Testing

Run tests from the `tests/` directory.