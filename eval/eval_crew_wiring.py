"""
End-to-end wiring test: drives the REAL pipeline -- real CrewAI Agent, Task,
Crew, output_pydantic conversion, and output guardrails -- with a
deterministic fake LLM in place of a live model. No API key and no network.

What this proves: the plumbing (schema field names, output_pydantic, kickoff,
guardrail routing, failure handling) is correct.

What this does NOT prove: anything about a real model's output quality. The
canned responses below are hand-written; whether a real model produces
summaries this good, or fabricates identifiers this way, can only be
measured with a live model (that is the LLM-as-Judge tier of the evaluation
pyramid described in the README, which this repo does not implement).

Run with:  python -m eval.eval_crew_wiring
"""

import json
import sys
from pathlib import Path
from typing import Any

from crewai import BaseLLM

from src.main import run_pipeline

SAMPLE_TRANSCRIPT = (
    Path(__file__).parent.parent / "sample_data" / "sample_transcript.txt"
).read_text(encoding="utf-8")

# A transcript with the same Medicare-like number as the sample but WITHOUT
# the embedded injection attempt, so the accept / hallucination paths can be
# tested without the injection escalation getting in the way.
CLEAN_TRANSCRIPT = (
    "Adviser: Can you confirm your Medicare number is still 2950 84715 1?\n"
    "Client: Yes that's still current. I'll take the new role in March.\n"
    "Adviser: Great. I'll send you the loan information by next Friday.\n"
)


class FakeLLM(BaseLLM):
    """Returns a canned response instead of calling a model."""

    canned: str = ""

    def call(self, messages, tools=None, callbacks=None, available_functions=None,
             from_task=None, from_agent=None, response_model=None) -> Any:
        return self.canned


class FailingLLM(BaseLLM):
    """Simulates a provider outage: every call raises."""

    def call(self, messages, tools=None, callbacks=None, available_functions=None,
             from_task=None, from_agent=None, response_model=None) -> Any:
        raise RuntimeError("simulated provider outage")


def canned(summary: str, **overrides: Any) -> str:
    payload = {
        "summary": summary,
        "action_items": [{"owner": "adviser", "description": "Send loan info", "due_date": "next Friday"}],
        "client_record_updates": ["New employment from March"],
        "flagged_for_review": False,
    }
    payload.update(overrides)
    return json.dumps(payload)


passed = 0
failed = 0


def check(condition: bool, label: str) -> None:
    global passed, failed
    if condition:
        print(f"  ✓ {label}")
        passed += 1
    else:
        print(f"  ✗ {label}")
        failed += 1


def main() -> int:
    print("=== Pipeline wiring (real CrewAI, fake model) ===\n")

    # 1. Well-formed output on a clean transcript -> ACCEPT
    r = run_pipeline(
        CLEAN_TRANSCRIPT,
        llm=FakeLLM(model="fake", canned=canned("Client confirmed details and will start a new role in March.")),
    )
    check(r.decision.action == "ACCEPT", "well-formed output on a clean transcript is ACCEPTed")

    # 2. Real identifier repeated back (it IS in the transcript) -> still ACCEPT
    r = run_pipeline(
        CLEAN_TRANSCRIPT,
        llm=FakeLLM(model="fake", canned=canned("Confirmed the client's Medicare number 2950 84715 1 is current.")),
    )
    check(r.decision.action == "ACCEPT", "an identifier that genuinely appears in the transcript is NOT flagged")

    # 3. Fabricated identifier -> FLAG_FOR_REVIEW with the hallucination reason
    r = run_pipeline(
        CLEAN_TRANSCRIPT,
        llm=FakeLLM(model="fake", canned=canned("Confirmed the client's Medicare number 1234 56789 0 is current.")),
    )
    check(r.decision.action == "FLAG_FOR_REVIEW", "a fabricated identifier is routed to FLAG_FOR_REVIEW")
    check(
        r.decision.action == "FLAG_FOR_REVIEW" and "not present in the source transcript" in r.decision.reason,
        "and the reason names the hallucinated identifier, not something else",
    )

    # 4. Clean model output, but the transcript contains an injection attempt
    #    -> escalated, never silently ACCEPTed
    r = run_pipeline(
        SAMPLE_TRANSCRIPT,
        llm=FakeLLM(model="fake", canned=canned("Client discussed a pay rise and super contributions.")),
    )
    check(r.injection.is_suspicious, "the sample transcript's injection attempt is detected")
    check(
        r.decision.action == "FLAG_FOR_REVIEW" and "prompt-injection" in r.decision.reason,
        "valid output on an injection-bearing transcript is escalated to FLAG_FOR_REVIEW",
    )

    # 5. The model flags its own uncertainty -> FLAG_FOR_REVIEW
    r = run_pipeline(
        CLEAN_TRANSCRIPT,
        llm=FakeLLM(model="fake", canned=canned("Ambiguous instruction from client.", flagged_for_review=True)),
    )
    check(r.decision.action == "FLAG_FOR_REVIEW", "the agent's own flagged_for_review signal is honoured")

    # 6. Failure paths: the pipeline must resolve to REJECT, never crash
    r = run_pipeline(CLEAN_TRANSCRIPT, llm=FailingLLM(model="fake"))
    check(
        r.decision.action == "REJECT" and "provider outage" in r.decision.reason,
        "a provider failure during the agent stage becomes REJECT, not a crash",
    )

    r = run_pipeline(
        CLEAN_TRANSCRIPT,
        llm=FakeLLM(model="fake", canned="Sure! Here's a friendly summary, no JSON at all."),
    )
    check(
        r.decision.action == "REJECT",
        "model output that cannot be converted to the schema becomes REJECT, not a crash",
    )

    print(f"\n{passed} passed, {failed} failed\n")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
