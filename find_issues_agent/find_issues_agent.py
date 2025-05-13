# find_issues_agent.py
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel, Field
from langchain_core.tools import tool
from find_issues_tools.audit_code_agent.audit_code_agent import audit_code_agent_tool
from find_issues_tools.audit_issue_agent.audit_issue_agent import audit_issue_agent_tool

prompt = '''
Your job it to listen to instructions from the user and run the appropriate tool
'''

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

agent = create_react_agent(
    model=llm,
    tools=[audit_code_agent_tool, audit_issue_agent_tool],
    prompt=prompt
)

if __name__ == "__main__":
    user_msg = "run the audit_code_agent_tool with the query 'find issues related to auth stuff'"
    response = agent.invoke({
        "messages": [{"role": "user", "content": user_msg}]
    })
    print(response)