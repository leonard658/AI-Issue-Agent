# push_issues_agent.py
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
import json
from pydantic import BaseModel, Field
from langchain_core.tools import tool
from push_issues_agent.tools.push_to_github import push_new_issue_to_github_tool, update_issue_on_github_tool
from find_issues_tools.audit_tools.semantic_query_vdb_tools import query_issues_tool
from pydantic_types.issue_schema import IssueList
from typing import Optional


class Summary(BaseModel):
    summary: str = Field(description="Summary of the issue that has been pushed/updated")
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
    response_format=Summary
)

class PushIssueAgentInputSchema(BaseModel):
    issue_description: str = Field(description="Description of the issue to work on")
    new_issue: bool = Field(description="Whether to create new issue (True) or work on existing one (False)")
    current_relevant_issues: Optional[IssueList] = Field(None, description="List of relevant issues")
@tool("push_issues_agent", args_schema=PushIssueAgentInputSchema, return_direct=False)
def push_issues_agent(input: PushIssueAgentInputSchema) -> str:
    """
    Agent that handles pushing issues to GitHub.
    """
    input_dict = input.model_dump()   # or .dict() depending on your Pydantic version
    # pretty-print the metadata as JSON:
    input_json = json.dumps(input_dict, indent=2)

    # build a single content string that includes instruction + metadata
    message = f"""
    Here is the information about the issue at hand:
    {input_json}
    """

    response = agent.invoke({
        "messages": [{"role": "user", "content": message}]
    }, debug=False)

    return response['structured_response'].summary



if __name__ == "__main__":
    test = PushIssueAgentInputSchema(
        issue_description="azure doest take files in through the responses api. It really should and will get fixed in the next update.",
        new_issue=False,
        current_relevant_issues=None
    )
    print(push_issues_agent(test))
    

