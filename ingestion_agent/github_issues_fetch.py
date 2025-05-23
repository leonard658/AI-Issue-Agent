import os
import requests
from dotenv import load_dotenv
from langchain_core.documents import Document

load_dotenv()

def load_issues(issues, slug):
    docs = []
    for entry in issues:
        metadata = {
            "author": entry["user"]["login"],
            "comments": entry["comments"],
            "title": entry["title"],
            "body": entry["body"],
            "labels": entry["labels"],
            "created_at": entry["created_at"],
            "updated_at": entry["updated_at"],
            "closed_at": entry["closed_at"] or "",
            "state": entry["state"],
            "number": entry["number"],
            "slug": slug,
        }
        data = entry["title"]
        if entry["body"]:
            data += ":\n" + entry["body"]
        # What this issue is going to get embedded based off of. title + body
        doc = Document(page_content=data, metadata=metadata)
        docs.append(doc)
    return docs

# Example usage:
# fetch_github_issues("leonard658","CustomLearnAI")
def fetch_github_issues(owner: str, repo: str) -> list[Document]:
    token = os.getenv("GITHUB_TOKEN")
    slug = f"{owner}/{repo}"
    repo_url = f"https://api.github.com/repos/{slug}/issues"
    headers = {"Authorization": f"Bearer {token}"}
    response = requests.get(repo_url, headers=headers)

    if response.status_code == 200:
        data = response.json()
        return load_issues(data, slug)

    return "Error: Unable to fetch issues."
#fetch_github_issues("leonard658","CustomLearnAI")

