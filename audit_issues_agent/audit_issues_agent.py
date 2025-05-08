# audit_issues_agent.py
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel, Field
from langchain_core.tools import tool
from audit_tools.query_pinecone_tools import query_issues_tool


llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

agent = create_react_agent(
    model=llm,
    tools=[query_issues_tool],
    prompt='''
    Your job is to audit issues passed to you by the find_issues_agent.
    You need to check if the issues are already reported in the vector db.
    Please provide a summary of the issues and if they are already reported in the vector db.
    '''
)

if __name__ == "__main__":
    user_msg = ""
    response = agent.invoke({
        "messages": [{"role": "user", "content": user_msg}]
    })
    print(response)