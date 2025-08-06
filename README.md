# AI Scrum Master - Azure Bot Setup

## Azure Bot Registration

The AI Scrum Master bot needs to be registered with the Azure Bot Service to work with Microsoft Teams. 

### Prerequisites

1. An Azure account
2. The Azure CLI installed
3. The Bot Framework CLI installed
4. A public HTTPS endpoint for your bot (e.g., using ngrok)

### Setup Instructions

1. **Run your bot locally or deploy it**

   Make sure your bot is running and accessible via HTTPS:
   ```bash
   python fastapi_teams_bot.py
   ```

2. **Create a tunnel to your local bot** (if testing locally)

   Use ngrok to create a secure tunnel:
   ```bash
   ngrok http 8080
   ```
   
   Note the HTTPS URL (e.g., https://6a96a73b9fdb.ngrok-free.app)

3. **Update the endpoint in your files**

   Edit the following files to use your ngrok URL:
   - `bot-manifest.json`
   - `azure-bot-registration.json`
   - `deploy-to-azure.sh`

4. **Deploy to Azure**

   Run the deployment script:
   ```bash
   chmod +x deploy-to-azure.sh
   ./deploy-to-azure.sh
   ```

5. **Add the bot to Microsoft Teams**

   Use one of these methods:
   - Upload the `bot-manifest.json` file using Teams App Studio
   - Install via the Teams store if you've published your bot

## Configuration

The bot uses the following environment variables:

- `MICROSOFT_APP_ID`: Your bot's App ID from Azure
- `MICROSOFT_APP_PASSWORD`: Your bot's App password/secret
- `PORT`: The port to run the bot on (default: 8080)

These should be set in your `.env` file.

## Testing Your Bot

1. Start a conversation with your bot in Teams
2. Type "start" to begin a standup
3. Follow the prompts to select a board and proceed with the standup

## Troubleshooting

- If your bot doesn't respond, check the logs to ensure it's receiving messages
- Verify your Azure Bot registration is correct and points to your endpoint
- Make sure your ngrok tunnel is running if testing locally
- Check that your `.env` file contains the correct credentials
