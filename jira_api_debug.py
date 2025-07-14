import os
import requests
from requests.auth import HTTPBasicAuth
from dotenv import load_dotenv

load_dotenv()

JIRA_URL = os.getenv("JIRA_URL")
JIRA_EMAIL = os.getenv("JIRA_EMAIL")
JIRA_API_TOKEN = os.getenv("JIRA_API_TOKEN")

if not all([JIRA_URL, JIRA_EMAIL, JIRA_API_TOKEN]):
    raise RuntimeError("JIRA_URL, JIRA_EMAIL, and JIRA_API_TOKEN must be set in the environment.")

# Ensure JIRA_EMAIL and JIRA_API_TOKEN are not None before using HTTPBasicAuth
if JIRA_EMAIL is None or JIRA_API_TOKEN is None:
    raise RuntimeError("JIRA_EMAIL and JIRA_API_TOKEN must be set in the environment.")
auth = HTTPBasicAuth(JIRA_EMAIL, JIRA_API_TOKEN)
headers = {"Accept": "application/json"}

def print_json(obj, indent=2):
    import json
    print(json.dumps(obj, indent=indent))

def check_boards():
    url = f"{JIRA_URL}/rest/agile/1.0/board"
    resp = requests.get(url, headers=headers, auth=auth)
    print(f"Boards API status: {resp.status_code}")
    boards = resp.json().get('values', [])
    for board in boards:
        print(f"Board: id={board.get('id')}, name={board.get('name')}")
    return boards

def check_sprints(board_id):
    url = f"{JIRA_URL}/rest/agile/1.0/board/{board_id}/sprint"
    resp = requests.get(url, headers=headers, auth=auth)
    print(f"Sprints API status: {resp.status_code}")
    sprints = resp.json().get('values', [])
    for sprint in sprints:
        print(f"Sprint: id={sprint.get('id')}, name={sprint.get('name')}, state={sprint.get('state')}")
    return sprints

def check_issues(sprint_id):
    url = f"{JIRA_URL}/rest/agile/1.0/sprint/{sprint_id}/issue"
    resp = requests.get(url, headers=headers, auth=auth)
    print(f"Issues API status: {resp.status_code}")
    issues = resp.json().get('issues', [])
    for issue in issues:
        fields = issue.get('fields', {})
        assignee = fields.get('assignee')
        assignee_name = assignee.get('displayName') if assignee else "Unassigned"
        print(f"Issue: key={issue.get('key')}, assignee={assignee_name}")
    return issues


if __name__ == "__main__":
    boards = check_boards()
    for board in boards:
        print(f"\n--- Checking sprints for board: {board.get('name')} (id={board.get('id')}) ---")
        sprints = check_sprints(board['id'])
        for sprint in sprints:
            print(f"\n--- Checking issues for sprint: {sprint.get('name')} (id={sprint.get('id')}) ---")
            check_issues(sprint['id'])
