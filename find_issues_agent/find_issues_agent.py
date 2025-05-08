# find_issues_agent.py
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel, Field
from langchain_core.tools import tool
from audit_tools.query_pinecone_tools import query_documents_tool, query_issues_tool



llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

agent = create_react_agent(
    model=llm,
    tools=[query_documents_tool, query_issues_tool],
    prompt="Your job it to identify issues in code."
)

if __name__ == "__main__":
    user_msg = "Ingest repo and issues from with owner-leonard658, repo-CustomLearnAi and branch-main"
    response = agent.invoke({
        "messages": [{"role": "user", "content": user_msg}]
    })
    print(response)