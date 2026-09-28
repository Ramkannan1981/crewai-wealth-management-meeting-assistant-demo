"""
Deterministic assertions against the guardrail layer -- no model call, no
API key, and no crewai import required (this file only imports the
Pydantic schema and the guardrail modules).

This is the bottom tier of the "evaluation pyramid" referenced in the
README: deterministic assertions -> LLM-as-Judge -> meta-evaluation. Only
the first tier is implemented here; it is the tier that should run on
every commit in CI, since it's fast and free.

Run with:  python -m eval.eval_guardrails
"""

import sys
from pathlib import Path

from src.guardrails.input_guardrails import detect_prompt_injection, redact_pii
from src.guardrails.output_guardrails import (
    detect_hallucinated_identifiers,
    run_output_guardrails,
    validate_structured_output,
)
from src.schemas import ActionItem, MeetingOutput

TRANSCRIPT_PATH = Path(__file__).parent.parent / "sample_data" / "sample_transcript.txt"

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
    transcript = TRANSCRIPT_PATH.read_text(encoding="utf-8")

    print("=== Input Guardrail Tests ===\n")

    result = redact_pii(transcript)
    found_types = {f.type for f in result.findings}
    check("EMAIL" in found_types, "detects the email address in the sample transcript")
    check("MEDICARE_LIKE" in found_types, "detects the Medicare-like number in the sample transcript")
    check("j.harper@example.com" not in result.redacted_text, "redacted text no longer contains the raw email")
    check("2950 84715 1" not in result.redacted_text, "redacted text no longer contains the raw Medicare-like number")

    injection = detect_prompt_injection(transcript)
    check(injection.is_suspicious, "detects the embedded prompt-injection attempt in the sample transcript")
    check(
        "instruction_override" in injection.matched_patterns,
        "specifically flags the 'ignore all previous instructions' pattern",
    )

    clean = detect_prompt_injection(
        "We discussed the client's super contributions and agreed to follow up next week."
    )
    check(not clean.is_suspicious, "does NOT flag ordinary transcript content as suspicious")

    print("\n=== Output Guardrail Tests ===\n")

    valid_output = MeetingOutput(
        summary="Client discussed increasing super contributions following a pay rise.",
        action_items=[
            ActionItem(
                owner="adviser",
                description="Send investment property loan information",
                due_date="next Friday",
            )
        ],
        client_record_updates=["Updated income following new role", "Added property purchase as near-term goal"],
        flagged_for_review=False,
    )
    check(
        validate_structured_output(valid_output.model_dump()).valid,
        "accepts a well-formed structured output",
    )
    check(
        not validate_structured_output({"action_items": "not a list"}).valid,
        "rejects output with a missing required field and a wrong type",
    )

    # A number that genuinely appears in the transcript must NOT be flagged.
    real_number_output = MeetingOutput(
        summary="Confirmed the client's Medicare number 2950 84715 1 is current."
    )
    check(
        not detect_hallucinated_identifiers(real_number_output, transcript).has_hallucinated_identifier,
        "does NOT flag an identifier that genuinely appeared in the source transcript",
    )

    # A different identifier-shaped number that was never in the transcript
    # is the actual hallucination case the guardrail exists to catch.
    fabricated_output = MeetingOutput(
        summary="Confirmed the client's Medicare number 1234 56789 0 is current."
    )
    check(
        detect_hallucinated_identifiers(fabricated_output, transcript).has_hallucinated_identifier,
        "flags an identifier-shaped value that does NOT appear anywhere in the source transcript",
    )
    decision = run_output_guardrails(fabricated_output.model_dump(), transcript)
    check(
        decision.action == "FLAG_FOR_REVIEW",
        "full guardrail chain routes a fabricated identifier to FLAG_FOR_REVIEW, not silent ACCEPT",
    )

    print(f"\n{passed} passed, {failed} failed\n")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
