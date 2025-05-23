# audit_issues_agent.py
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel, Field
from langchain_core.tools import tool


prompt = '''

'''

llm = ChatOpenAI(model="gpt-4o-mini", temperature=.8)

agent = create_react_agent(
    model=llm,
    tools=[],
    prompt=prompt,
)

class AuditIssueAgentToolInput(BaseModel):
    query: str = Field(description="Description of the issue to generate a summary for / query to find relevant, currently documented issues")
@tool("audit_issue_agent_tool", args_schema=AuditIssueAgentToolInput, return_direct=False)
def audit_issue_agent_tool(query: str) -> str:
    """
    Retrieve and curate a report on the currently documented issues for any query. 
    This can be used to see if an issue is documented sufficiently already, documented poorly or not documented at all.
    The output from this function should be used to drive decisions on whether to continue to look into issues, document them, or any ambiguous question in between.
    This summary provides concise, high-quality context to help analyze or resolve specific questions related to currently documented issues.
    """
    response = agent.invoke({
        "messages": [{"role": "user", "content": query}]
    }, debug=False)
    return response['messages'][-1].content

if __name__ == "__main__":
    example_find_issues_msg = "We are working through an issue related to azure."
    print(audit_issue_agent_tool(example_find_issues_msg))
