# unified-llm-client

Provider-agnostic LLM client library.

## Setup
```bash
cd unified-llm-client
python -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
python -m pip install anthropic google-genai
```

## Tests
```bash
. .venv/bin/activate
pytest -q
```

Provider adapter tests require `anthropic` and `google-genai`. If those packages are missing, tests will fail on import.
