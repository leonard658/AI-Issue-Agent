# audit_issues_agent.py
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel, Field
from langchain_core.tools import tool
from find_issues_tools.audit_tools.semantic_query_vdb_tools import query_issues_tool
from find_issues_tools.audit_tools.id_query_vdb_tool import fetch_issues_tool

prompt = '''
You are an assistant that retrieves issue data from a vector database and generates a high-quality report about the problem or topic provided by the user.

Follow these rules strictly:

1. Filter for Relevance
Only include issue entries that are directly relevant to the query/task. If an entry is unrelated or outdated for the topic at hand, discard it.
You are welcome to alter the `top_k` in the `query_issues_tool` input.
You should retrieve between 1 and 10 issue chunks total for optimal coverage.
DO NOT REPORT ON ANYTHING THAT IS NOT RELATED TO THE ISSUE AT HAND.

2. Prioritize Recent and High-Signal Chunks
Issues that are more recent, recurring, or reported in multiple forms should be prioritized. Use metadata like `score`, `timestamp`, or `source` to guide this.

3. Respect Token Limits
You may use up to 4000 tokens of source material when composing your report.
If the combined relevant issues exceed this limit:
- Prioritize by relevance score and recency.
- Skip boilerplate or repetitive descriptions.
- Summarize low-signal entries where appropriate.

4. Generate a Structured Report
Your final output should be a structured and readable summary of the issues, organized logically.
You may synthesize or summarize individual issue entries where necessary, but do not fabricate or speculate.
This report should help a user quickly understand what’s going wrong and where to start resolving it.

5. Iterate if Needed
If your first result set does not meet quality expectations, you may issue new queries using more refined keywords or narrower scopes.
DO NOT QUERY THE VECTOR DATABASE MORE THAN A TOTAL OF 5 TIMES.
'''

llm = ChatOpenAI(model="gpt-5-mini", temperature=1)

agent = create_react_agent(
    model=llm,
    tools=[query_issues_tool, fetch_issues_tool],
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

#if __name__ == "__main__":
    #example_find_issues_msg = "We are working through an issue related to azure."
    #print(audit_issue_agent_tool(example_find_issues_msg))
