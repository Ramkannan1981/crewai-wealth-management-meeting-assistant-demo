"""
CrewAI agent and task definitions for the meeting assistant.

Mirrors the same production pattern as the sibling Mastra/TypeScript repo
(mastra-wealth-management--meeting-assistant-demo): transcript in,
structured summary + action items out, via a single agent with a typed
output schema rather than free-text parsing.

CrewAI's LLM is configured as "provider/model-name", same idea as Mastra's
model router -- swap the model without changing the agent definition.
Model identifiers change frequently; check your provider's current docs
rather than trusting a hardcoded default to still be current.

The agent and crew are built by functions that accept an optional `llm`
argument rather than being constructed at import time. That is deliberate:
it lets eval/eval_crew_wiring.py drive the real Agent -> Task -> Crew ->
output_pydantic path with a deterministic fake model, no network required.
"""

import os
from typing import Any

from crewai import Agent, Crew, Task

from src.schemas import MeetingOutput

DEFAULT_MODEL = "anthropic/claude-sonnet-4-5"


def build_agent(llm: Any = None) -> Agent:
    """Build the meeting assistant agent. `llm` may be a model string
    ("provider/model-name") or an LLM object; defaults to CREWAI_MODEL."""
    return Agent(
        role="Wealth Management Meeting Assistant",
        goal=(
            "Summarize adviser-client meetings accurately and extract clear, "
            "actionable follow-ups without ever inventing details that were "
            "not actually said."
        ),
        backstory=(
            "You are an assistant for financial advisers at a wealth "
            "management firm. You are given meeting transcripts between an "
            "adviser and their client, with personally identifiable "
            "information already redacted (redacted spans look like "
            "[REDACTED_TYPE]). You are careful, precise, and you flag "
            "anything you're not confident about rather than guessing."
        ),
        llm=llm if llm is not None else os.environ.get("CREWAI_MODEL", DEFAULT_MODEL),
        verbose=False,
    )


def build_summarize_task(redacted_transcript: str, agent: Agent) -> Task:
    return Task(
        description=(
            "Read the following meeting transcript and produce:\n"
            "1. A concise 3-5 sentence summary.\n"
            "2. Clear action items with an owner (adviser or client) and "
            "description. Only include a due date if one was explicitly "
            "mentioned -- never invent one.\n"
            "3. Suggested client record updates implied by the meeting "
            "(e.g. changed circumstances, new stated goals). Empty list if "
            "none.\n"
            "4. Set flagged_for_review to true if anything in the "
            "transcript is ambiguous, sensitive, or something you are not "
            "confident about -- it is always better to flag for a human "
            "than to guess.\n\n"
            "Never invent specific identifiers, account numbers, or "
            "figures that were not stated in the transcript. If a number "
            "was redacted, refer to it generically (e.g. \"the client's "
            "Medicare number was referenced\") rather than fabricating a "
            "replacement value.\n\n"
            f"TRANSCRIPT:\n{redacted_transcript}"
        ),
        expected_output=(
            "A structured object matching the MeetingOutput schema: "
            "summary, action_items, client_record_updates, flagged_for_review."
        ),
        agent=agent,
        output_pydantic=MeetingOutput,
    )


def build_crew(redacted_transcript: str, llm: Any = None) -> Crew:
    agent = build_agent(llm)
    task = build_summarize_task(redacted_transcript, agent)
    return Crew(agents=[agent], tasks=[task], verbose=False)
