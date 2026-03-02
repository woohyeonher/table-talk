==================================================

# TableTalk

==================================================

TableTalk is a structured dish recommendation chatbot built with FastAPI and Vertex AI.

Users describe what they want to eat, and the bot selects exactly one dish from a fixed catalog using a strict decision protocol:

DECISION: ASK
DECISION: RECOMMEND
DECISION: REFUSE

All responses follow a controlled format and enforce clear scope boundaries, safety handling, and post-generation validation.

==================================================
## Prerequisites

You must have Google Cloud set up with Vertex AI enabled.

Create a file named `.env` in the project root directory with:

VERTEXAI_PROJECT=your-project-id
VERTEXAI_LOCATION=us-central1

You must also be authenticated with gcloud.
==================================================
## Installation

Clone the repository and create a virtual environment.

```bash
python -m venv .venv
.\.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```
==================================================
## Running the App

From the project root directory, run:

uv run python app.py

Then open the following URL in your browser:

http://localhost:8000

==================================================
## API Endpoints

- `GET /`
Serves the TableTalk web UI.

- `POST /chat`
Accepts a user message and returns a structured decision response.

- `POST /clear`
Clears the current session history.

==================================================
## Prompt Design

The system prompt includes:

- Clear role and persona definition

- Positive scope definition (what the bot can do)

- 3+ out-of-scope categories using positive framing

- Few-shot examples

- Cuisine handling logic

- An uncertainty escape hatch

- Hard structural rules

- Normalization logic

Response formats:

If more information is needed:

DECISION: ASK
Q1: ...
Q2: ...

If recommending:

DECISION: RECOMMEND
DISH: <dish_id>
WHY: <1–3 sentences>

If out-of-scope:

DECISION: REFUSE
WHY: ...
ALT: ...

==================================================
## Safety and Python Backstop

In addition to prompt-level constraints, a Python post-generation backstop:

- Verifies a valid DECISION field

- Enforces required structure per decision type

- Detects malformed or unsafe outputs

- Falls back to safe refusal when necessary

This ensures structured outputs even if the model drifts.

==================================================
## Evaluation Harness

The `evals/ directory` contains pytest-based evaluations using Model-as-a-Judge.

Files:

`evals/test_golden.py`

- 20 golden test cases

- 10 in-domain

- 5 out-of-scope

- 5 adversarial

`evals/test_rubric.py`

- Rubric-based grading against weighted quality criteria

`evals/test_rules.py`

- Deterministic protocol checks using regex and keyword validation

To run all evaluations:

```bash
uv run pytest -v -s
```

Note:
Evaluations make live LLM calls to both the bot and a judge model.
They require network access and will incur API costs.

==================================================
