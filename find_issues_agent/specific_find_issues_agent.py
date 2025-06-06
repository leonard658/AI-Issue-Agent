# find_issues_agent.py
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel, Field
from langchain_core.tools import tool
from find_issues_tools.audit_code_agent.audit_code_agent import audit_code_agent_tool
from find_issues_tools.audit_issue_agent.audit_issue_agent import audit_issue_agent_tool
from pydantic_types.document_schema import DocumentsChunkSchema
from pydantic_types.to_json_str import to_json_str
from push_issues_agent.push_issues_agent import push_issues_agent
from typing import Optional
from find_issues_tools.internet_search.tavily_search import basic_tavily_search
from find_issues_tools.internet_search.gpt_researcher import gpt_researcher_tool

class Summary(BaseModel):
    summary: str = Field(description="Summary of the all the issues that have been addressed in the current chunk")
prompt = '''
You are an expert software auditor, tasked with **systematically going through potential issues that have been preliminarily identified by another agent**.

**Instructions:**
- Do an in-depth review of the issue at hand.
    - This mean in-depth. You should use a variety of tools to confirm initial reviews of the issue.
- Your focus should be pretty narrow in this search.
- To do this you may **Call tools as needed** to help analyze or clarify the issue (e.g., for deeper audit, semantic queries, or to check related context).
- Once you have sufficiently thought through the current issue and wish to push it to github, call the push_issues_agent
    - This agent will return a summary of what it did with the issue pushing to github

**Tools available for better understanding:**  
- `audit_code_agent_tool`: for in-depth code audit and static analysis.
- `audit_issue_agent_tool`: for granular investigation of specific issues.
- `basic_tavily_search`: for basic internet search questions.
- `gpt_researcher_tool`: for in depth report generation about a topic that uses the internet.

**Only push issues that are relevant to the current chunk.**
You are methodical, objective, and prioritize clear, actionable output.
'''

llm = ChatOpenAI(model="gpt-4.1", temperature=0.2)

agent = create_react_agent(
    model=llm,
    tools=[audit_code_agent_tool, audit_issue_agent_tool, basic_tavily_search, gpt_researcher_tool, push_issues_agent],
    prompt=prompt,
    response_format=Summary
)

class SpecificFindIssueAgentInputSchema(BaseModel):
    code_chunk: DocumentsChunkSchema = Field(description="Current code chunk to work on")
    issue_description: str = Field(description="Description of the issue to work on in the current code chunk")
    new_issue: bool = Field(description="Whether this is an issue that has been identified (True) or not (False)")
    relevant_code: Optional[list[DocumentsChunkSchema]] = Field(None, description="List of relevant code chunks to current chunk")
@tool("specific_find_issues_agent", args_schema=SpecificFindIssueAgentInputSchema, return_direct=False)
def specific_find_issues_agent(code_chunk: DocumentsChunkSchema, issue_description: str, new_issue: bool, relevant_code: Optional[list[DocumentsChunkSchema]]) -> str:
    """
    Run the find issues agent tool with the given DocumentChunk
    """
    input = SpecificFindIssueAgentInputSchema(
        code_chunk=code_chunk,
        issue_description=issue_description,
        new_issue=new_issue,
        relevant_code=relevant_code
    )
    
    doc_str = to_json_str(input)
    
    message = f'''
    Here is the issue and relevant background to take a deep dive into:
    {doc_str}
    '''
    
    response = agent.invoke({
        "messages": [{"role": "user", "content": message}]
    }, debug=False)
    #print("DocChunk Processed: ", doc_chunk.metadata.id)
    
    return response['structured_response'].summary


#if __name__ == "__main__":
    #user_msg = "run the audit_code_agent_tool with the query 'find issues related to auth stuff'"
    #response = agent.invoke({
    #    "messages": [{"role": "user", "content": user_msg}]
    #})
    #print(response)