# find_issues_agent.py
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel, Field
from langchain_core.tools import tool
from find_issues_tools.audit_code_agent.audit_code_agent import audit_code_agent_tool
from find_issues_tools.audit_issue_agent.audit_issue_agent import audit_issue_agent_tool
from pydantic_types.document_schema import DocumentsChunkSchema

prompt = '''
Your job find any issues in the code that are passed to you.
Generate a report of the issues you find.
You have many tools at your disposal to help you find issues.
'''

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

agent = create_react_agent(
    model=llm,
    tools=[audit_code_agent_tool, audit_issue_agent_tool],
    prompt=prompt
)

def run_find_issues_agent(DocChunk: DocumentsChunkSchema) -> str:
    """
    Run the find issues agent tool with the given DocumentChunk
    """
    #response = agent.invoke({
    #    "messages": [{"role": "user", "content": str(DocChunk)}]
    #}, debug=False)
    print("DocChunk Processed: ", DocChunk.metadata.id)
    return ""
    #return response['messages'][-1].content


#if __name__ == "__main__":
    #user_msg = "run the audit_code_agent_tool with the query 'find issues related to auth stuff'"
    #response = agent.invoke({
    #    "messages": [{"role": "user", "content": user_msg}]
    #})
    #print(response)