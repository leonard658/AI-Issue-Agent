import os
from typing import Optional
from pydantic import BaseModel, Field
import requests
from dotenv import load_dotenv
from langchain_core.tools import tool

load_dotenv()

class GitHubIssueError(Exception):
    """Raised when GitHub’s API responds with a non-2xx status code."""


# ── Create a new issue ────────────────────────────────────────────
class PushNewIssueToolInput(BaseModel):
    slug: str = Field(description="Owner/repo slug for the GitHub repository")
    title: str = Field(description="Title of the issue")
    body: str = Field(description="Body of the issue")
    labels: list[str] = Field(default=[], description="Labels to attach to the issue")
@tool("push_new_issue_to_github_tool", args_schema=PushNewIssueToolInput, return_direct=False)
def push_new_issue_to_github_tool(
    slug: str,               # "owner/repo"
    title: str,
    body: str,
    labels: list[str] = []
) -> dict:
    """
    Create a new GitHub issue.
    """
    token = os.getenv("GITHUB_TOKEN")
    if not token:
        raise RuntimeError("GITHUB_TOKEN must be set in the environment")
    
    # labels should always have the issue_agent unique label
    labels.append("issue agent")

    url = f"https://api.github.com/repos/{slug}/issues"
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept":        "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    payload = {"title": title, "body": body, "labels": labels}

    r = requests.post(url, json=payload, headers=headers)
    if not r.ok:
        raise GitHubIssueError(f"[{r.status_code}] {r.text}")

    data = r.json()
    return {
        "number": data["number"],
        "url":    data["html_url"],
        "title":  data["title"],
        "state":  data["state"],
    }


# ── Update an existing issue ──────────────────────────────────────
class UpdateIssueToolInput(BaseModel):
    slug: str = Field(description="Owner/repo slug for the GitHub repository")
    issue_number: int = Field(description="Issue number to update")
    title: Optional[str] = Field(None, description="New title for the issue")
    body: Optional[str] = Field(None, description="New body for the issue")
    labels: Optional[list[str]] = Field([], description="New labels for the issue")
@tool("update_issue_on_github_tool", args_schema=UpdateIssueToolInput, return_direct=False)
def update_issue_on_github_tool(
    slug: str,               # "owner/repo"
    issue_number: int,
    title: Optional[str] = None,
    body: Optional[str] = None,
    labels: Optional[list[str]] = [],
) -> dict:
    """
    Update an existing GitHub issue.
    Only slug and issue_number are required; title, body, and labels are optional.
    """
    token = os.getenv("GITHUB_TOKEN")
    if not token:
        raise RuntimeError("GITHUB_TOKEN must be set in the environment")

    # build payload with only the fields you want to change
    payload: dict = {}
    if title is not None:
        payload["title"] = title
    if body is not None:
        payload["body"] = body

    # labels should always have the issue_agent unique label
    labels.append("issue agent")
    payload["labels"] = labels

    if not payload:
        raise ValueError("At least one of title, body, or labels must be provided to update an issue")

    url = f"https://api.github.com/repos/{slug}/issues/{issue_number}"
    headers = {
        "Authorization":        f"Bearer {token}",
        "Accept":               "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }

    r = requests.patch(url, json=payload, headers=headers)
    if not r.ok:
        raise GitHubIssueError(f"[{r.status_code}] {r.text}")

    data = r.json()

    return {
        "number": data["number"],
        "url":    data["html_url"],
        "title":  data["title"],
        "state":  data["state"],
    }

#print(push_new_issue_to_github("leonard658/CustomLearnAI", "Test issue", "This is a test issue", ["bug"]))
#print(update_issue_on_github("leonard658/CustomLearnAI", 15, title="NEW TITLEEEE", body="NEW BODY", labels=["bug"]))