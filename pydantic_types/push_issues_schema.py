from typing import TypedDict
from typing import Optional

class PushIssuesSchema(TypedDict):
    title: str
    body: str
    labels: list[str]
    slug: str
    short_summary: str
    issue_number: int | None
    