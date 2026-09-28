"""
Output guardrails.

Two distinct checks, deliberately kept separate:

1. Structured-output validation -- does the model's output actually match
   the contract in schemas.py? A model can return plausible-looking JSON
   that doesn't conform (wrong types, missing fields); this is the check
   that catches that before it reaches a caller.

2. Hallucinated-identifier detection -- the specific, higher-stakes check:
   does the output contain something that LOOKS like a Medicare number,
   TFN, or similar identifier that was NOT present anywhere in the original
   transcript? If so, the model has almost certainly generated
   (hallucinated) a plausible-looking identifier rather than reporting one
   that was actually said in the meeting -- which in a wealth management
   context is exactly the failure mode that must never reach a client
   record silently.
"""

import json
import re
from dataclasses import dataclass
from typing import Literal, Union

from pydantic import ValidationError

from src.schemas import MeetingOutput

# Same identifier-shaped patterns as the input guardrail, reused here for a
# different purpose: not redaction, but "does this appear in the output but
# NOT in the source transcript."
_IDENTIFIER_PATTERNS = [
    re.compile(r"\b\d{4}\s?\d{5}\s?\d{1}\b"),  # Medicare-like
    re.compile(r"\b\d{3}\s?\d{3}\s?\d{2,3}\b"),  # TFN-like
]


@dataclass
class ValidationResult:
    valid: bool
    errors: list[str]
    data: MeetingOutput | None


def validate_structured_output(raw_output: dict) -> ValidationResult:
    try:
        data = MeetingOutput.model_validate(raw_output)
        return ValidationResult(valid=True, errors=[], data=data)
    except ValidationError as e:
        errors = [f"{'.'.join(str(p) for p in err['loc'])}: {err['msg']}" for err in e.errors()]
        return ValidationResult(valid=False, errors=errors, data=None)


@dataclass
class HallucinatedIdentifierResult:
    has_hallucinated_identifier: bool
    suspect_values: list[str]


def detect_hallucinated_identifiers(
    output: MeetingOutput, original_transcript: str
) -> HallucinatedIdentifierResult:
    """
    Compares identifier-shaped strings found in the agent's output against
    the original (unredacted) transcript. Anything present in the output
    but absent from the source transcript is flagged as a likely
    hallucination, not a real value the client or adviser actually stated.

    NOTE: this intentionally runs against the *original* transcript, not
    the PII-redacted version -- redaction happens on the way into the model
    (see input_guardrails.py); this check happens on the way out, comparing
    against ground truth.
    """
    output_text = output.model_dump_json()
    suspect_values: list[str] = []
    normalized_source = re.sub(r"\s", "", original_transcript)

    for pattern in _IDENTIFIER_PATTERNS:
        for match in pattern.findall(output_text):
            normalized_match = re.sub(r"\s", "", match)
            if normalized_match not in normalized_source:
                suspect_values.append(match)

    return HallucinatedIdentifierResult(
        has_hallucinated_identifier=len(suspect_values) > 0,
        suspect_values=suspect_values,
    )


@dataclass
class AcceptDecision:
    action: Literal["ACCEPT"]
    data: MeetingOutput


@dataclass
class FlagForReviewDecision:
    action: Literal["FLAG_FOR_REVIEW"]
    data: MeetingOutput
    reason: str


@dataclass
class RejectDecision:
    action: Literal["REJECT"]
    reason: str


OutputDecision = Union[AcceptDecision, FlagForReviewDecision, RejectDecision]


def run_output_guardrails(raw_output: dict, original_transcript: str) -> OutputDecision:
    """Runs the full output guardrail chain and returns a final decision."""
    validation = validate_structured_output(raw_output)
    if not validation.valid or validation.data is None:
        return RejectDecision(
            action="REJECT",
            reason=f"Structured output validation failed: {'; '.join(validation.errors)}",
        )

    hallucination_check = detect_hallucinated_identifiers(validation.data, original_transcript)
    if hallucination_check.has_hallucinated_identifier:
        return FlagForReviewDecision(
            action="FLAG_FOR_REVIEW",
            data=validation.data,
            reason=(
                f"Output contains identifier-shaped value(s) not present in the source "
                f"transcript: {len(hallucination_check.suspect_values)} suspect value(s) found. "
                f"This is treated as a likely hallucination, not a confirmed client detail."
            ),
        )

    if validation.data.flagged_for_review:
        return FlagForReviewDecision(
            action="FLAG_FOR_REVIEW",
            data=validation.data,
            reason="Agent itself flagged this output for human review.",
        )

    return AcceptDecision(action="ACCEPT", data=validation.data)
