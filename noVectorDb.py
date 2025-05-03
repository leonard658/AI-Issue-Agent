from dotenv import load_dotenv
import os

from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
from github import fetch_github_issues

load_dotenv()

# 1) Grab your GitHub issues once at startup
owner = "leonard658"
repo = "CustomLearnAI"
issues = fetch_github_issues(owner, repo)
# issues is a list of Document(page_content=…, metadata={…})

# 2) Turn them into one big string
issues_text = "\n\n".join(
    f"Issue #{i+1} by {doc.metadata['author']} ({doc.metadata['created_at']}):\n"
    f"{doc.page_content}"
    for i, doc in enumerate(issues)
)

# 3) Initialize your chat model
llm = ChatOpenAI()

# 4) System prompt (you can tweak this)
system = SystemMessage(content=(
    "You are an AI assistant that answers questions about a GitHub repo’s issues. "
    "Below are all the open issues; reference them directly when you reply."
))

# 5) Interactive Q&A loop
while True:
    q = input("Ask a question about GitHub issues (q to quit): ")
    if q.lower() == "q":
        break

    human = HumanMessage(content=(
        f"Here are all the issues:\n\n{issues_text}\n\n"
        f"User question: {q}"
    ))

    # Send messages to the model
    answer = llm.predict_messages([system, human])
    print("\n" + answer.content + "\n")


