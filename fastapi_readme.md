# AI Scrum Master with FastAPI for MS Teams

This application provides a FastAPI interface for the AI Scrum Master Bot, designed to integrate with Microsoft Teams. The bot automates daily standups by facilitating conversations with team members about their progress, blockers, and plans.

## Architecture:

This solution consists of:

1. **FastAPI Backend**: Provides endpoints for the bot to communicate with
2. **Core Scrum Master Logic**: Reused from the original Streamlit implementation
3. **MS Teams Integration**: Connect the bot to Teams using the Bot Framework

## Setup & Configuration

### Prerequisites

- Python 3.8+
- JIRA account with API access
- MongoDB instance
- Pinecone account (for semantic search)
- OpenAI API Key (for completions and embeddings)
- Microsoft Teams Bot registration

### Environment Variables

Create a `.env` file with the following variables:

```
# JIRA Configuration
JIRA_BASE_URL=https://your-jira-instance.atlassian.net
JIRA_USER_EMAIL=your.email@example.com
JIRA_API_TOKEN=your_jira_api_token

# MongoDB Configuration
MONGO_URI=mongodb://username:password@hostname:port/database

# OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key

# Pinecone Configuration
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_ENVIRONMENT=your_pinecone_env
PINECONE_INDEX_NAME=your_pinecone_index_name

# Microsoft Bot Configuration
MicrosoftAppId=your_ms_app_id
MicrosoftAppPassword=your_ms_app_password

# Server Configuration
PORT=8000
```

### Installation

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Start the FastAPI server:
   ```bash
   python fastapi_teams_bot.py
   ```

## API Endpoints

### Core Endpoints

- `GET /` - Health check endpoint
- `GET /boards` - List all available JIRA boards
- `POST /webhook` - Main webhook endpoint for MS Teams Bot Framework

### Session Flow Endpoints

These endpoints manage the conversation flow:

- `POST /start` - Start a new bot session
- `POST /select_board` - Select a JIRA board
- `POST /select_user` - Select a JIRA user and start the standup
- `POST /message` - Process messages in an ongoing standup

## MS Teams Integration

1. Create a bot in [Bot Framework](https://dev.botframework.com/)
2. Configure the messaging endpoint to point to your `/webhook` endpoint
3. Add the bot to your Teams channel

### Configuring the Bot in Teams

1. Register your bot in the Azure portal
2. Generate an App ID and password
3. Add these credentials to your `.env` file
4. Deploy the FastAPI application to a publicly accessible URL
5. Configure the messaging endpoint in the Bot Framework registration
6. Install the bot in Teams using the App Studio

## Deploying to Production

For production deployment, consider:

- Using a production ASGI server like Uvicorn behind Nginx
- Setting up proper authentication for the webhook endpoint
- Implementing persistent storage for bot sessions (e.g., Redis)
- Setting up monitoring and logging
