# agent_base.py
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent


llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

agent = create_react_agent(
    model=llm,
    tools=[],
    prompt="You are the base agent."
)

if __name__ == "__main__":
    user_msg = "What tools are available?"
    response = agent.invoke({
        "messages": [{"role": "user", "content": user_msg}]
    })