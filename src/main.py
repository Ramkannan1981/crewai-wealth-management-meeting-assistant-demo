"""
End-to-end pipeline:

  raw transcript
    -> [INPUT GUARDRAILS]  PII redaction, prompt-injection scan
    -> [AGENT]             CrewAI agent, structured output (Pydantic)
    -> [OUTPUT GUARDRAILS] schema validation, hallucinated-identifier check
    -> final decision (ACCEPT / FLAG_FOR_REVIEW / REJECT)

Mirrors the sibling Mastra/TypeScript repo -- same contract, same
guardrails, different orchestration framework.

The pipeline always resolves to an explicit decision. In particular, if the
agent stage itself fails (CrewAI raises when a model's output can't be
converted to the schema), that is caught and becomes a REJECT rather than a
crash -- "visible failure, not silent" applies to the pipeline as a whole,
not just to the guardrail functions.

Run with:  python -m src.main
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from src.guardrails.input_guardrails import (
    PiiFinding,
    PromptInjectionResult,
    detect_prompt_injection,
    redact_pii,
)
from src.guardrails.output_guardrails import (
    AcceptDecision,
    FlagForReviewDecision,
    OutputDecision,
    RejectDecision,
    run_output_guardrails,
)

TRANSCRIPT_PATH = Path(__file__).parent.parent / "sample_data" / "sample_transcript.txt"


@dataclass
class PipelineResult:
    injection: PromptInjectionResult
    pii_findings: list[PiiFinding]
    decision: OutputDecision


def run_pipeline(raw_transcript: str, llm: Any = None) -> PipelineResult:
    """Run all three stages and return the final decision.

    `llm` is optional and exists so tests can inject a deterministic fake
    model; in normal use the model comes from CREWAI_MODEL.
    """
    # Stage 1: input guardrails
    injection = detect_prompt_injection(raw_transcript)
    redaction = redact_pii(raw_transcript)

    # Stage 2: agent. Imported here, not at module top, so the no-key path
    # in main() never depends on crewai or a provider extra being installed.
    from src.agents import build_crew

    try:
        result = build_crew(redaction.redacted_text, llm=llm).kickoff()
    except Exception as exc:  # pipeline boundary: any agent failure -> REJECT
        return PipelineResult(
            injection,
            redaction.findings,
            RejectDecision(
                action="REJECT",
                reason=f"Agent stage failed: {type(exc).__name__}: {exc}",
            ),
        )

    if result.pydantic is None:
        decision: OutputDecision = RejectDecision(
            action="REJECT", reason="Agent returned no structured output."
        )
    else:
        # Stage 3: output guardrails. The hallucination check runs against the
        # ORIGINAL transcript, not the redacted one -- see output_guardrails.py.
        decision = run_output_guardrails(result.pydantic.model_dump(), raw_transcript)

    # A transcript containing injection patterns is never silently ACCEPTed,
    # even if the output itself validated cleanly.
    if injection.is_suspicious and isinstance(decision, AcceptDecision):
        decision = FlagForReviewDecision(
            action="FLAG_FOR_REVIEW",
            data=decision.data,
            reason=(
                "Transcript contained prompt-injection patterns "
                f"({', '.join(injection.matched_patterns)}); flagged for human "
                "review even though the output validated cleanly."
            ),
        )

    return PipelineResult(injection, redaction.findings, decision)


def _print_input_stage(raw_transcript: str) -> None:
    print("=== STAGE 1: Input Guardrails ===\n")
    injection = detect_prompt_injection(raw_transcript)
    if injection.is_suspicious:
        print(f"⚠ Prompt-injection patterns detected: {', '.join(injection.matched_patterns)}\n")
    else:
        print("✓ No prompt-injection patterns detected.\n")

    print("PII findings (counts only, no raw values logged):")
    findings = redact_pii(raw_transcript).findings
    if not findings:
        print("  none")
    for finding in findings:
        print(f"  {finding.type}: {finding.count}")
    print()


def main() -> None:
    load_dotenv()
    raw_transcript = TRANSCRIPT_PATH.read_text(encoding="utf-8")

    _print_input_stage(raw_transcript)

    print("=== STAGE 2 + 3: Agent (CrewAI) and Output Guardrails ===\n")

    if not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("OPENAI_API_KEY")):
        print(
            "No model API key found in the environment (ANTHROPIC_API_KEY / OPENAI_API_KEY).\n"
            "Skipping the live agent call -- see README for setup.\n"
            "To exercise the full pipeline without a key, run:\n"
            "  python -m eval.eval_crew_wiring\n"
        )
        return

    decision = run_pipeline(raw_transcript).decision
    print(f"Decision: {decision.action}")
    if isinstance(decision, RejectDecision):
        print(f"Reason: {decision.reason}")
    elif isinstance(decision, FlagForReviewDecision):
        print(f"Reason: {decision.reason}")
        print("Output (requires human review before use):")
        print(decision.data.model_dump_json(indent=2))
    else:
        print("Output (safe to return to caller):")
        print(decision.data.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
