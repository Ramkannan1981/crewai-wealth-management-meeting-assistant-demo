"""
Input guardrails.

These run BEFORE the transcript is ever sent to the model. This mirrors
the real production split: what you protect against on the way IN
(sensitive data reaching the model, or the model being manipulated by
content embedded in the transcript) is a different problem from what you
protect against on the way OUT (see output_guardrails.py).

Deliberately implemented as pure, synchronous functions with no model
calls and no dependency on crewai itself -- these are cheap, deterministic,
and fully unit-testable without an API key or the crewai package installed.
See eval/eval_guardrails.py.
"""

import re
from dataclasses import dataclass, field


@dataclass
class PiiFinding:
    type: str
    # The matched text is intentionally NOT stored here -- findings should
    # be safe to log without re-exposing the PII they describe.
    count: int


@dataclass
class PiiRedactionResult:
    redacted_text: str
    findings: list[PiiFinding] = field(default_factory=list)


# Patterns are deliberately conservative (prefer false positives over false
# negatives for a guardrail) and are illustrative, not exhaustive. A
# production system would layer a proper PII-detection model or service
# (e.g. AWS Comprehend, Presidio) on top of pattern matching like this.
_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("EMAIL", re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")),
    ("PHONE", re.compile(r"\b(?:\+?61|0)[2-478](?:[ -]?\d){8}\b")),
    # Australian Medicare card numbers: 10 digits, sometimes formatted with spaces
    ("MEDICARE_LIKE", re.compile(r"\b\d{4}\s?\d{5}\s?\d{1}\b")),
    # Australian Tax File Numbers: 8-9 digits
    ("TFN_LIKE", re.compile(r"\b\d{3}\s?\d{3}\s?\d{2,3}\b")),
    ("CREDIT_CARD_LIKE", re.compile(r"\b(?:\d[ -]?){13,16}\b")),
]


def redact_pii(raw_text: str) -> PiiRedactionResult:
    """
    Redacts likely-PII from transcript text before it reaches the model or
    any log line. Returns both the redacted text and a count-only summary
    of what was found, so callers can decide whether to flag the document
    for review without ever having the raw PII pass back through
    application code.
    """
    redacted_text = raw_text
    findings: list[PiiFinding] = []

    for pii_type, pattern in _PATTERNS:
        matches = pattern.findall(redacted_text)
        if matches:
            findings.append(PiiFinding(type=pii_type, count=len(matches)))
            redacted_text = pattern.sub(f"[REDACTED_{pii_type}]", redacted_text)

    return PiiRedactionResult(redacted_text=redacted_text, findings=findings)


@dataclass
class PromptInjectionResult:
    is_suspicious: bool
    matched_patterns: list[str] = field(default_factory=list)


# Heuristic patterns for content embedded in a transcript that is attempting
# to manipulate the agent's instructions, rather than being genuine meeting
# content. Real systems typically combine this kind of heuristic layer with
# a model-based classifier; this demo shows the deterministic layer only.
_INJECTION_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("instruction_override", re.compile(r"ignore (all|any|the) (previous|prior|above) instructions?", re.IGNORECASE)),
    ("role_override", re.compile(r"you are now", re.IGNORECASE)),
    ("system_prompt_probe", re.compile(r"(reveal|show|print) (your|the) (system prompt|instructions)", re.IGNORECASE)),
    ("exfiltration_attempt", re.compile(r"forward (this|all) (data|information|transcript) to", re.IGNORECASE)),
]


def detect_prompt_injection(text: str) -> PromptInjectionResult:
    """
    Scans transcript content for patterns suggesting the text is attempting
    to inject instructions to the agent, rather than being genuine
    transcript content. This does not block the transcript -- it flags it
    for the flagged_for_review output field, consistent with a "visible
    failure, not silent" design (see README's Design Notes section).
    """
    matched = [label for label, pattern in _INJECTION_PATTERNS if pattern.search(text)]
    return PromptInjectionResult(is_suspicious=len(matched) > 0, matched_patterns=matched)
