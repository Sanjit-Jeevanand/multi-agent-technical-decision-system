from openai import OpenAI
from typing import Any, Dict, Optional

_client: Optional[OpenAI] = None


def get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI()
    return _client


def call_responses_api(
    *,
    model: str,
    system_prompt: str,
    user_prompt: str,
    reasoning_effort: str = "none",
    verbosity: str = "low",
) -> str:
    """
    Unified Responses API call.
    Returns raw text output (JSON string expected).
    """

    client = get_client()

    response = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        reasoning={"effort": reasoning_effort},
        text={"verbosity": verbosity},
    )

    # Always extract text safely
    return response.output_text