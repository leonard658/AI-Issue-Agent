# audit_code_agent.py
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel, Field
from langchain_core.tools import tool
from audit_tools.query_pinecone_tools import query_documents_tool


llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

agent = create_react_agent(
    model=llm,
    tools=[query_documents_tool],
    prompt='''
    Your job it to pull code from a repo and scrub it and make sure it is relevant to the find_issues_agent that you report to. 
    If you get code from the vector db but it is not relevant to the find_issues_agent, then you should not report it, and get the relevant code from the vector db.
    If you get code from the vector db that is relevant to the find_issues_agent but incomplete, then please find the other relevant chunks from the vector db.
    If you get code from the vector db that is relevant to the find_issues_agent and complete, then please report it to the find_issues_agent.
    '''
)

if __name__ == "__main__":
    user_msg = ""
    response = agent.invoke({
        "messages": [{"role": "user", "content": user_msg}]
    })
