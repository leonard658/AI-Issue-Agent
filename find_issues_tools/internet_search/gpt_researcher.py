# research_tool.py

import asyncio
import json
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from langchain_core.tools import tool
from gpt_researcher import GPTResearcher

load_dotenv()

async def _run_research(query: str, report_type: str):
    """
    Internal async helper that runs GPTResearcher and gathers all outputs.
    """
    researcher = GPTResearcher(query, report_type, verbose=False)
    research_result = await researcher.conduct_research()
    report = await researcher.write_report()#custom_prompt="Answer in short, 2 paragraphs max without citations.")

    # Fetch additional metadata from the researcher object
    #research_context = researcher.get_research_context()
    research_costs = researcher.get_costs()
    #research_images = researcher.get_research_images()
    #research_sources = researcher.get_research_sources()

    return report, research_costs#, research_conetext, research_images, research_sources

class ResearchToolInput(BaseModel):
    query: str = Field(..., description="The natural‐language question to research")
    report_type: str = Field(default='research_report', description="Type of report (e.g., 'research_report', 'summary', etc.)")
@tool("gpt_researcher_tool", args_schema=ResearchToolInput, return_direct=False)
def gpt_researcher_tool(query: str, report_type: str = 'research_report') -> str:
    """
    Researcher and reporter tool if basic search is not enough and in depth report is needed
    Returns a JSON string containing all of:
      - report
    """
    report, costs = asyncio.run(_run_research(query, report_type))

    print(costs)
    output = {
        "report": report,
        #"research_context": context,
        #"research_costs": costs,
        #"research_images": images,
        #"research_sources": sources,
    }

    # Return a JSON‐encoded string so downstream LangGraph nodes can parse it.
    return json.dumps(output)

if __name__ == "__main__":
    # Invoke the tool and get its output string
    result = gpt_researcher_tool.invoke({
        "query": "Who should i bet on in the nba finals this year"
    })

    # Write that string into a file (e.g. “output.txt”)
    with open("output.txt", "w", encoding="utf-8") as f:
        f.write(result)
    print("Result written to output.txt")