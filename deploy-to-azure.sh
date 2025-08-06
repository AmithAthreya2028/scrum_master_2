#!/bin/bash

# Script to deploy the AI Scrum Master bot to Azure Bot Service
# Prerequisites: Azure CLI and Bot Framework CLI installed
# Run: chmod +x deploy-to-azure.sh && ./deploy-to-azure.sh

# Load environment variables
source .env

# Set up variables
RESOURCE_GROUP="ai-scrum-master-resources"
LOCATION="eastus"
BOT_NAME="ai-scrum-master"
MESSAGING_ENDPOINT="https://6a96a73b9fdb.ngrok-free.app/webhook"
APP_ID="$MICROSOFT_APP_ID"
APP_PASSWORD="$MICROSOFT_APP_PASSWORD"

# Login to Azure if needed
echo "Checking Azure login status..."
az account show > /dev/null 2>&1
if [ $? -ne 0 ]; then
    echo "Please login to Azure..."
    az login
fi

# Create resource group if it doesn't exist
echo "Creating resource group if it doesn't exist..."
az group create --name "$RESOURCE_GROUP" --location "$LOCATION" --output none

# Create/update bot registration
echo "Registering bot with Azure Bot Service..."
az bot create \
    --resource-group "$RESOURCE_GROUP" \
    --name "$BOT_NAME" \
    --kind "registration" \
    --endpoint "$MESSAGING_ENDPOINT" \
    --msa-app-id "$APP_ID" \
    --password "$APP_PASSWORD" \
    --output none

# Enable MS Teams channel
echo "Enabling Microsoft Teams channel..."
az bot msteams create \
    --resource-group "$RESOURCE_GROUP" \
    --name "$BOT_NAME" \
    --output none

# Update the manifest
echo "Updating bot manifest..."
sed -i "s/\"botId\": \".*\"/\"botId\": \"$APP_ID\"/" bot-manifest.json
sed -i "s/\"id\": \".*\",/\"id\": \"$APP_ID\",/" bot-manifest.json
sed -i "s/\"id\": \".*\"/\"id\": \"$APP_ID\"/" bot-manifest.json
sed -i "s|https://.*ngrok-free.app|$MESSAGING_ENDPOINT|g" bot-manifest.json

echo "Bot registration complete! Your bot is now registered with Azure Bot Service."
echo "To add it to Microsoft Teams:"
echo "1. Go to the Bot Framework portal: https://dev.botframework.com/bots"
echo "2. Select your bot and click 'Channels'"
echo "3. Click the Microsoft Teams icon and follow the instructions"
echo "4. Or use the Teams App Studio to upload the manifest file directly"
echo ""
echo "Messaging endpoint: $MESSAGING_ENDPOINT"
echo "App ID: $APP_ID"
echo "Bot name: $BOT_NAME"
