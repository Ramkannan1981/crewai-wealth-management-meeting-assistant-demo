# CrewAI Meeting Assistant Demo

A small, self-contained agentic AI pipeline: a meeting transcript goes in, a
structured summary + action items come out — with input and output
guardrails wired around a [CrewAI](https://crewai.com) agent.

This is the **CrewAI counterpart** to my
[Mastra version of the same demo](https://github.com/Ramkannan1981/mastra-wealth-management--meeting-assistant-demo).
Same output contract, same guardrail logic, different orchestration
framework — built side by side so the two are directly comparable.

## What this is (and isn't)

My production agentic-AI work is built on **Mastra** (TypeScript, AWS
Bedrock). I have not shipped CrewAI to production. This repo is me applying
the same production patterns in CrewAI to make the framework comparison
concrete — it is a **from-scratch reference implementation**, not a copy of
anything proprietary. It contains no internal code, no internal
architecture, and no real client data; the sample transcript is synthetic.

It demonstrates:

- **Input guardrails** — PII redaction and prompt-injection detection,
  applied *before* the transcript reaches the model.
- **Structured output via a typed contract** (Pydantic) rather than
  free-text parsing.
- **Output guardrails**, including **hallucinated-identifier detection**:
  if the output contains an identifier-shaped value (e.g. something that
  looks like a Medicare number) that was *not* in the source transcript, it
  is treated as a likely fabrication and routed to human review.
- **Explicit decisions, never a silent pass-through or a crash** — the
  pipeline always resolves to `ACCEPT`, `FLAG_FOR_REVIEW`, or `REJECT`.
  That includes agent-stage failures (provider outage, output that can't be
  converted to the schema) and transcripts containing prompt-injection
  patterns, which are escalated rather than accepted even when the output
  itself validates.

## Architecture

```
raw transcript
  │
  ▼
[INPUT GUARDRAILS]   PII redaction + prompt-injection scan
  ▼
[AGENT]              CrewAI Agent → Task (output_pydantic=MeetingOutput) → Crew
  ▼
[OUTPUT GUARDRAILS]  1. schema validation
  │                  2. hallucinated-identifier check (output vs. source)
  │                  3. the agent's own flagged_for_review signal
  ▼
ACCEPT / FLAG_FOR_REVIEW / REJECT
  (a transcript with injection patterns is never silently ACCEPTed)
```

## Project structure

```
src/
  schemas.py                    Pydantic output contract
  agents.py                     CrewAI Agent / Task / Crew builders (LLM injectable)
  guardrails/input_guardrails.py    PII redaction, injection detection
  guardrails/output_guardrails.py   Validation, hallucination check, decision chain
  main.py                       run_pipeline() + CLI entry point
eval/
  eval_guardrails.py            Deterministic guardrail assertions (12)
  eval_crew_wiring.py           End-to-end pipeline test with a fake LLM (9)
sample_data/
  sample_transcript.txt         Synthetic transcript exercising every guardrail path
```

The guardrail modules and schema depend only on Pydantic — not on CrewAI.

## Setup

Tested on Python 3.12.

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

> Provider support in CrewAI is an optional extra. `requirements.txt`
> installs `crewai[anthropic]` for the default model. A plain
> `pip install crewai` fails with an `ImportError` the first time an agent
> is built. If you switch `CREWAI_MODEL` to another provider, install that
> provider's extra instead (e.g. `crewai[openai]`).

## Testing

No API key or network access is needed for either suite:

```bash
python -m eval.eval_guardrails    # 12 deterministic assertions on the guardrails
python -m eval.eval_crew_wiring   # 9 assertions driving the real pipeline
```

`eval_crew_wiring` runs the **real** `run_pipeline()` — real CrewAI Agent,
Task, Crew, and `output_pydantic` conversion — with a deterministic fake
model in place of a live one. It covers accept, fabricated-identifier,
injection-escalation, self-flagged, provider-failure, and unparseable-output
paths. (You'll see a few `Error executing listener ... simulated provider
outage` lines while it runs; that is CrewAI logging the failure the test
deliberately injects.)

I also checked that these tests can actually fail: temporarily disabling the
hallucination check made the relevant assertions fail in both suites.

**What is not verified here:** any behaviour of a real model. The canned
responses in the tests are hand-written, so they show the *plumbing* is
correct — not that a live model writes good summaries or fabricates
identifiers in the way the tests assume. I have not run this against a live
provider, and the default model string (`anthropic/claude-sonnet-4-5`) has
not been validated against a live API; model names change often, so check
your provider's docs and set `CREWAI_MODEL` if needed. Measuring real-model
quality is the LLM-as-Judge tier of an evaluation pyramid
(deterministic assertions → LLM-as-Judge → meta-evaluation); this repo
implements only the first tier.

## Running the pipeline

```bash
python -m src.main                # no key: runs Stage 1, then explains what's needed
```

With a key:

```bash
cp .env.example .env              # set ANTHROPIC_API_KEY (or OPENAI_API_KEY + CREWAI_MODEL)
python -m src.main
```

The sample transcript deliberately contains a Medicare-like number, an
email address, and an embedded injection attempt ("*ignore all previous
instructions…*"), so a run exercises every stage. Expect the final decision
on that transcript to be `FLAG_FOR_REVIEW` because of the injection attempt,
even when the model's output is otherwise clean.

## Mastra vs CrewAI: what differed in practice

Observations from building both:

- **Structured output.** Mastra takes a Zod schema per call
  (`structuredOutput: { schema }`, result on `.object`). CrewAI attaches a
  Pydantic model to the *Task* (`output_pydantic=`, result on `.pydantic`).
- **Failure behaviour.** CrewAI's `kickoff()` *raises* (`ConverterError`)
  when model output can't be converted to the schema, rather than returning
  an empty result — so the pipeline needs an explicit boundary that turns
  that into `REJECT`. `run_pipeline()` has one.
- **Setup.** Provider support is an optional extra (`crewai[anthropic]`);
  see Setup above.
- **Testability.** CrewAI's `BaseLLM` has a single abstract method
  (`call`), which made a deterministic fake model straightforward. That is
  why `agents.py` builds the agent from a function taking an optional `llm`
  instead of at import time.

## Design notes

- **Input vs output are separate concerns.** What's *in* the data (PII) and
  whether the *output* can be trusted are different problems, implemented as
  separate modules.
- **The hallucination check runs against the original transcript, not the
  redacted one.** Redaction controls what reaches the model; hallucination
  detection verifies the model's output against ground truth.
- **PII findings are logged as counts and types only, never raw values.**
- **This is a reference implementation.** Regex-based PII detection
  demonstrates the pattern; a production system should layer a real
  PII-detection model or managed service on top rather than rely on regex.

## License

MIT
