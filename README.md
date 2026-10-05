# Self Dependent AI

A local-first coding-agent prototype. Give it one coding goal; it can inspect the selected workspace, ask a configured language model what to do next, and recover from tool errors within a fixed step and retry budget. File changes and test runs always require your approval.

This is an orchestrator around an OpenAI-compatible chat-completions API, not a newly trained foundation model. It has no account integrations, persistent memory, or unrestricted shell access.

A newer collaboration layer adds a safe routing mechanism for specialist work, such as research, coding, and verification. Model handoffs are explicit and capability-based; they never grant a model new access or credentials.

## Requirements

- Python 3.10 or newer
- An API key for an OpenAI-compatible provider

## Setup

From this project directory, create and activate a virtual environment, then install the package:

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -e .
```

On Windows, the generated `self-dependent-ai` console script is not always available from a live PowerShell session if PATH is not refreshed. The most reliable entry point is the module form below, and the repository also includes a PowerShell wrapper at `run_self_dependent_ai.ps1` for the same purpose:

```powershell
python -m self_dependent_ai.cli --help
./run_self_dependent_ai.ps1 --help
```

Set an API key and model for the current PowerShell session. The key is read from the environment and is not saved by this application:

```powershell
$env:OPENAI_API_KEY = "your-key"
$env:OPENAI_BASE_URL = "https://api.openai.com/v1"
$env:OPENAI_MODEL = "gpt-4o-mini"
```

You can also define a specialist registry for research, coding, verification, and review in a single JSON payload:

```powershell
$env:SELF_DEPENDENT_PROVIDERS = '{"research_a":{"api_key":"your-key","base_url":"https://api.openai.com/v1","model":"gpt-4o-mini","capabilities":["research"]},"coding_a":{"api_key":"your-key","base_url":"https://api.openai.com/v1","model":"gpt-4o-mini","capabilities":["coding"]},"verification_a":{"api_key":"your-key","base_url":"https://api.openai.com/v1","model":"gpt-4o-mini","capabilities":["verification"]},"review_a":{"api_key":"your-key","base_url":"https://api.openai.com/v1","model":"gpt-4o-mini","capabilities":["review"]}}'
```

A ready-to-copy example is included in `.env.example`.

## Run

Select a project directory as the agent's workspace and pass a single goal:

```powershell
python -m self_dependent_ai.cli "Fix the failing unit tests in this project" --workspace .
# or
./run_self_dependent_ai.ps1 "Fix the failing unit tests in this project" --workspace .
```

For a collaborative review mode that validates the provider output through a second model before finishing:

```powershell
python -m self_dependent_ai.cli "Research the alpha API before coding" --workspace . --collaborative
# or
./run_self_dependent_ai.ps1 "Research the alpha API before coding" --workspace . --collaborative
```

The agent can list and read non-sensitive files, propose file edits, and run the project's unittest suite. It asks before applying an edit or running tests. Declining an action does not execute it. It never executes arbitrary model-provided shell commands.

The agent limits each run to 12 model steps and allows at most 2 consecutive recovery failures. Treat test execution as running code from the selected project; approve it only when you trust that code. Do not point it at folders containing secrets or data you do not want sent to your model provider.

## Tests

```powershell
python -m unittest discover -s tests -v
```

## Publish Welcome to the Future

The astrology app is prepared for deployment on Render. The web process uses Gunicorn, serves `/healthz` for health checks, applies defensive response headers, and marks submitted reading responses as non-cacheable. Birth details are processed in memory for a reading and are not written to a database or file. The hosting provider will still receive normal request metadata such as IP address; review its privacy terms before publishing.

To make it public, this project must first be pushed to a GitHub repository that you control:

1. Create a private or public repository on GitHub. A public code repository is not required for a public website; private is preferable if you do not intend to publish the source.
2. From this project directory, connect that repository as `origin`, then push the project files.
3. In Render, choose **New** then **Blueprint**, connect the repository, and deploy the included `render.yaml`.
4. After the deployment becomes healthy, use the HTTPS `onrender.com` URL shown by Render. Keep the repository private unless you explicitly want the source code public.

Do not commit `.venv`, `.env`, API credentials, or personal birth data. The existing `.gitignore` excludes the virtual environment and `.env`; task data is local and should also remain untracked.

## Current limits

The model's response format is validated as JSON but model output can still be wrong. This prototype does not offer account-to-account access, multi-model collaboration, durable memory, or guarantees that generated changes are correct. Review proposed changes and test results before relying on them.
