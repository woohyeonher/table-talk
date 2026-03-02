"""Shared fixtures for Project 1 restaurant recommender evals.

Provides three core helpers:
  - `get_review`: sends a user prompt to the restaurant chatbot, returns its response.
  - `judge_with_golden`: judge compares output to an expected reference (rating 1-10).
  - `judge_with_rubric`: judge grades output against a rubric (rating 1-10).

Note:
We keep the function name `get_review` so existing tests can import it unchanged.
"""

import json
import re
import sys
from pathlib import Path

from litellm import completion

# Add parent directory so we can import app.py
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# The app is expected to expose:
# - MODEL: litellm model string
# - build_initial_messages(): returns initial chat messages with system prompt, etc.
from app import MODEL, build_initial_messages


# --- Bot (system under test) ---

JUDGE_MODEL = "vertex_ai/gemini-2.0-flash"


def get_review(text: str) -> str:
    """Send a prompt to the restaurant chatbot and return its response text."""
    messages = build_initial_messages()
    messages.append({"role": "user", "content": text})

    # Use deterministic generation for eval stability.
    result = completion(
        model=MODEL,
        messages=messages,
        temperature=0,
    )
    return result.choices[0].message.content


# --- Judge helpers ---

JUDGE_SYSTEM_GOLDEN = """\
You are an expert evaluator for a restaurant dish recommendation chatbot.

You will be given:
- a user prompt
- a reference response (golden)
- a generated response

Rate the overall quality of the generated response from 1 to 10 by comparing
to the reference response. Consider:
- correctness with respect to user constraints (allergies, dislikes, budget, etc.)
- appropriateness of DECISION (ASK vs RECOMMEND vs REFUSE)
- adherence to the required output format:
  - includes "DECISION: ASK|RECOMMEND|REFUSE"
  - if DECISION is RECOMMEND, includes "DISH: <pretty dish name>"
  - if DECISION is ASK, includes at least one question line "Q1: ..."
  - may include "ALT: ..." alternatives
- helpfulness and clarity
- safety: does not comply with unsafe or out-of-scope requests

Return ONLY a valid JSON object with a single key "rating" whose value is an
integer between 1 and 10.

Example response:
{
  "rating": 7
}"""

JUDGE_SYSTEM_RUBRIC = """\
You are an expert evaluator for a restaurant dish recommendation chatbot.

You will be given:
- a user prompt
- a generated response
- a list of rubrics (with weights)

Rate the overall quality of the generated response from 1 to 10 based on how
well it satisfies the rubrics. Consider all rubrics holistically.

Also check format adherence:
- includes "DECISION: ASK|RECOMMEND|REFUSE"
- if DECISION is RECOMMEND, includes "DISH: <pretty dish name>"
- if DECISION is ASK, includes at least one "Q1: ..."
- optionally includes "ALT: ..."

Return ONLY a valid JSON object with a single key "rating" whose value is an
integer between 1 and 10.

Example response:
{
  "rating": 7
}"""


def judge_with_golden(prompt: str, reference: str, response: str) -> int:
    """Judge a response against a golden reference. Returns rating 1-10."""
    user_msg = (
        "Compare the generated response to the reference response and rate it."
        f"\n\n<prompt>\n{prompt}\n</prompt>"
        f"\n\n<reference_response>\n{reference}\n</reference_response>"
        f"\n\n<generated_response>\n{response}\n</generated_response>"
    )
    result = completion(
        model=JUDGE_MODEL,
        temperature=0,
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM_GOLDEN},
            {"role": "user", "content": user_msg},
        ],
    )
    return _parse_rating(result.choices[0].message.content)


def judge_with_rubric(prompt: str, response: str, rubric: str) -> int:
    """Judge a response against a rubric. Returns rating 1-10."""
    user_msg = (
        "Grade the response against the rubrics and rate it."
        f"\n\n<prompt>\n{prompt}\n</prompt>"
        f"\n\n<response>\n{response}\n</response>"
        f"\n\n<rubrics>\n{rubric}\n</rubrics>"
    )
    result = completion(
        model=JUDGE_MODEL,
        temperature=0,
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM_RUBRIC},
            {"role": "user", "content": user_msg},
        ],
    )
    return _parse_rating(result.choices[0].message.content)



def _extract_first_json_object(text: str) -> str:
    """Return the first JSON object substring found in text using brace matching."""
    start = text.find("{")
    if start == -1:
        raise ValueError(f"Could not find '{{' in judge response: {text!r}")

    depth = 0
    in_string = False
    escape = False

    for i in range(start, len(text)):
        ch = text[i]

        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
            continue

        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]

    raise ValueError(f"Unbalanced braces in judge response: {text!r}")


def _parse_rating(text: str) -> int:
    """Extract the integer rating from a judge response that contains JSON."""
    if not text:
        raise ValueError("Judge returned empty response.")

    json_str = _extract_first_json_object(text)
    obj = json.loads(json_str)

    if "rating" not in obj:
        raise ValueError(f'Judge JSON missing "rating": {obj!r}')

    rating = int(obj["rating"])
    if rating < 1 or rating > 10:
        raise ValueError(f"Rating out of range 1-10: {rating}")

    return rating