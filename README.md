# redteam-harness

An offline-first LangGraph harness for evaluating a chatbot deployment that you own or control. It generates one adversarial test per strategy category, sends it to a target, judges the response with a separate model call, mutates the state, and stops on budget, severity threshold, or exhausted categories.

> **Safety:** Use this only against systems you own or are authorized to test. Do not publish successful attack prompts or raw transcripts in a public repository or README. The normal report contains aggregate results only; raw attempts are written to the private debug output path.

## Setup

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
pytest
```

## Run the harness

This works out of the box with a free local Ollama model (`llama3.2`), though it is slower. For faster, higher-quality results, set `OPENAI_API_KEY` and `provider: openai` in `config.yaml`.

The default target is the local Ollama model `llama3.2`; pull it with `ollama pull llama3.2` first. The target's system rules are set by `target_spec`, and `target_model` selects its local model. Set `target_mode: mock` for a deterministic, fast smoke test. The mock target is intended for tests and smoke testing, not behavior evaluation.

For an authorized HTTP target, configure `target_mode: http`, `target_url`, `target_headers`, `request_body_template`, and `response_text_path` in `config.yaml`. The request template substitutes `{prompt}` recursively; response paths use dot notation and numeric list indexes. HTTP mode prints a warning and requires the explicit `--i-have-permission` flag:

```powershell
redteam-harness --config config.yaml --i-have-permission
```

Requests default to a 30-second timeout and two total attempts. Failed requests are recorded as unevaluated attempts rather than terminating the run.

Run the attacker, judge, and target model calls with:

```powershell
redteam-harness --config config.yaml
```

`report.txt` is the aggregate report. `debug_attempts.jsonl` contains raw prompts and responses for private review and is ignored by Git.
