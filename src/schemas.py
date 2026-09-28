"""
Structured output contract for the meeting assistant crew.

This is the CrewAI/Python counterpart to the Mastra/TypeScript version of
this same demo (see the sibling repo:
mastra-wealth-management--meeting-assistant-demo). Same contract, same
guardrail logic, different orchestration framework -- deliberately, so the
two repos are directly comparable.

This mirrors a real production pattern: the agent is not free-texting a
summary, it is required to return data matching this shape. That is what
makes the output-side guardrails in guardrails/output_guardrails.py
possible -- you can only validate and redact fields you know exist.
"""

from typing import Optional
from pydantic import BaseModel, Field


class ActionItem(BaseModel):
    owner: str = Field(description="Who is responsible for this action item")
    description: str = Field(description="What needs to be done")
    due_date: Optional[str] = Field(
        default=None, description="Due date if mentioned, otherwise null"
    )


class MeetingOutput(BaseModel):
    summary: str = Field(description="A concise summary of the meeting, 3-5 sentences")
    action_items: list[ActionItem] = Field(
        default_factory=list, description="Extracted action items from the meeting"
    )
    client_record_updates: list[str] = Field(
        default_factory=list,
        description=(
            "Suggested updates to the client's record based on the meeting "
            "(e.g. changed circumstances, new goals). Empty list if none."
        ),
    )
    flagged_for_review: bool = Field(
        default=False,
        description=(
            "True if anything in the transcript needs human review before "
            "this output is used (e.g. ambiguous instruction, sensitive "
            "topic, low confidence)"
        ),
    )
