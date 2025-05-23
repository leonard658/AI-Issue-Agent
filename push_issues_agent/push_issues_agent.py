# audit_issues_agent.py
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel, Field
from langchain_core.tools import tool
from pydantic_types.push_issues_schema import PushIssuesSchema
from push_issues_agent.tools.push_to_github import push_new_issue_to_github_tool, update_issue_on_github_tool
from find_issues_tools.audit_tools.semantic_query_vdb_tools import query_issues_tool

prompt = '''
You are a GitHub issue generator.
You receive raw notes about a code issue and must return one GitHub issue in Markdown.

You can:
* Queries the repo’s vector database of existing issues (using the query_issues_tool) to detect duplicates; if it is a duplicate, just edit the existing issue.
* You have tools to create a new issue or update an existing one.
* Uses a short, compelling title (≤ 60 chars).
* States the problem objectively; don’t guess root cause.
* For potential runtime errors, include exact steps or commands to reproduce, plus expected vs. actual results.
* List stack traces, file paths, or code snippets as needed (fenced blocks).
* Add environment info (OS, language version, branch/commit).
* Output must follow the section order shown below—nothing more, nothing less.
* Return a short summary of what you did.

---

Example output:
Title: NullPointerException in UserService.create()

Body:
````markdown
**Issue description:**
A clear and concise description of what the problem is. Ex. I'm always frustrated when [...]

**Ideal solution:**
A clear and concise description of what you want to happen.

**Potential alternatives:**
A clear and concise description of any alternative solutions or features you've considered.

**Additional context**
Add any other context or screenshots about the feature request here.

```

'''

llm = ChatOpenAI(model="gpt-4o-mini", temperature=1)

agent = create_react_agent(
    model=llm,
    tools=[push_new_issue_to_github_tool, update_issue_on_github_tool, query_issues_tool],
    prompt=prompt,
)

def push_issues_agent(info: str) -> str:
    """
    Agent that handles pushing issues to GitHub.
    """
    response = agent.invoke({
        "messages": [{"role": "user", "content": info}]
    }, debug=False)

    return "Summary of pushed issue:\n" + response['messages'][-1].content



if __name__ == "__main__":
    msg = "We are working through an issue related to azure."
    response = agent.invoke({
        "messages": [{"role": "user", "content": msg}]
    }, debug=False)
    print(response['structured_response'])
