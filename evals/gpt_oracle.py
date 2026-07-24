"""GPT-5.5 gateway client + LLM-as-judge, used only for *grading* fact-QA.

The PoC puts real facts directly in the model's context (no retrieval), so GPT
is no longer a retriever. It is used only to grade free-form fact-QA answers
semantically (crediting paraphrases/aliases that exact/substring matching
misses). `openai` is imported lazily so this module imports offline.
"""

from __future__ import annotations

import os
import time

from organizer.store import normalize

DEFAULT_BASE_URL = "https://tfy.promptlens.trilogy.com/v1"
DEFAULT_MODEL = "openai-group/gpt-5.5"


def default_model() -> str:
    return os.environ.get("POC_GPT_MODEL", DEFAULT_MODEL)


def clean_value(text: str) -> str:
    """First non-empty line, stripped of surrounding quotes and a trailing period."""
    if not text:
        return ""
    line = next((ln.strip() for ln in text.splitlines() if ln.strip()), "")
    return line.strip().strip('"').strip("'").strip().removesuffix(".").strip()


def answer_matches(pred: str, possible_answers) -> bool:
    """String-match fallback: alias-tolerant substring (either direction)."""
    if not pred:
        return False
    np_ = normalize(pred)
    for a in possible_answers or []:
        na = normalize(a)
        if na and (na == np_ or na in np_ or np_ in na):
            return True
    return False


class GatewayClient:
    """OpenAI-compatible client for the TrueFoundry gateway (reads
    OPENAI_API_KEY / OPENAI_BASE_URL from the env)."""

    def __init__(self, model: str | None = None, base_url: str | None = None,
                 api_key: str | None = None, max_tokens: int = 64,
                 temperature: float = 0.0, max_retries: int = 4):
        from openai import OpenAI

        self.model = model or default_model()
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.max_retries = max_retries
        self._client = OpenAI(
            api_key=api_key or os.environ.get("OPENAI_API_KEY"),
            base_url=base_url or os.environ.get("OPENAI_BASE_URL", DEFAULT_BASE_URL),
        )

    def chat(self, system: str, user: str, max_tokens: int | None = None) -> str:
        last_err: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                resp = self._client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "system", "content": system},
                              {"role": "user", "content": user}],
                    temperature=self.temperature,
                    max_tokens=max_tokens or self.max_tokens,
                )
                return resp.choices[0].message.content or ""
            except Exception as err:  # noqa: BLE001 - surface after retries
                last_err = err
                time.sleep(min(2 ** attempt, 8))
        raise RuntimeError(f"gateway call failed after {self.max_retries} retries: {last_err}")

    def answer(self, question: str) -> str:
        return clean_value(self.chat(
            "Answer with ONLY the factual value (a name/place/date/short phrase), "
            "no sentence or extra words.", question))

    def smoke(self, question: str = "What is the capital of France?") -> str:
        return self.answer(question)


_JUDGE_SYSTEM = (
    "You grade short answers. Given the question, the reference (correct) answer, "
    "and a candidate answer, decide whether the candidate is correct — accept "
    "paraphrases, aliases, and equivalent phrasings (e.g. 'soccer' == 'association "
    "football', 'US' == 'United States'). Reply with exactly one word: YES or NO."
)


def judge_answer(client, question: str, reference: str, candidate: str) -> bool:
    """LLM-as-judge semantic correctness. Empty candidate is always NO."""
    if not candidate or not candidate.strip():
        return False
    user = (f"Question: {question}\nReference answer: {reference}\n"
            f"Candidate answer: {candidate}\nIs the candidate correct? YES or NO.")
    return client.chat(_JUDGE_SYSTEM, user, max_tokens=4).strip().upper().startswith("Y")
