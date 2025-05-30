# find_issues_agent.py
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel, Field
from langchain_core.tools import tool
from find_issues_tools.audit_code_agent.audit_code_agent import audit_code_agent_tool
from find_issues_tools.audit_issue_agent.audit_issue_agent import audit_issue_agent_tool
from find_issues_agent.specific_find_issues_agent import specific_find_issues_agent
from pydantic_types.document_schema import DocumentsChunkSchema
from pydantic_types.to_json_str import to_json_str


class Summary(BaseModel):
    summary: str = Field(description="Summary of the all the issues that hve been addressed in the current chunk")
prompt = '''
You are an expert software auditor, tasked with **systematically finding issues in provided code chunks**.

**Instructions:**
- Quickly review the provided code chunk.
- Identify any potential issues, including but not limited to:
    - Bugs or logic errors
    - Bad or unclear variable/function names
    - Poor documentation or missing comments
    - Performance bottlenecks
    - Security vulnerabilities
    - Code smells (duplication, complexity, etc.)
    - Violations of standard style guides or best practices
- To do this you may **Call tools as needed** to help analyze or clarify the issue (e.g., for deeper audit, semantic queries, or to check related context).
- Once you find a potential issue, call the specific_find_issues_agent 
    - The specific_find_issues_agent will taker a deeper look into the issue you provide and document issues as neeed.
    - It will then return a summary of what it did which you will use later 

**Tools available for better understanding:**  
- `audit_code_agent_tool`: for in-depth code audit and static analysis.
- `audit_issue_agent_tool`: for granular investigation of specific issues.

**Only report issues you have evidence for in the current chunk. If additional context is required, clearly state what’s missing.
For code in other chunks, you will have a chance to look at it later so don't worry about addressing issues in them right now.**

You are methodical, objective, and prioritize clear, actionable output.
'''

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

agent = create_react_agent(
    model=llm,
    tools=[audit_code_agent_tool, audit_issue_agent_tool, specific_find_issues_agent],
    prompt=prompt,
    response_format=Summary
)

#class PushIssueAgentInputSchema(BaseModel):
#    doc_chunk: DocumentsChunkSchema = Field(description="Chunk of code to work on ")
#@tool("push_issues_agent", args_schema=PushIssueAgentInputSchema, return_direct=False)
def broad_find_issues_agent(doc_chunk: DocumentsChunkSchema) -> str:
    """
    Run the find issues agent tool with the given DocumentChunk
    """
    doc_str = to_json_str(doc_chunk)
    
    message = f'''
    Here is the chunk of the document to focus on right now:
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