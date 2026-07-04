"""LLM-based question paraphrasing for benchmark hardening (plan step П.2).

LLM-generated benchmark questions lexically mirror their answer sentence, so
seed retrieval finds the answer by keyword overlap and HR saturates. This
module rewrites questions to ask for the same fact with different vocabulary,
verifying the result with ``content_token_overlap`` and retrying with an
explicit forbidden-word list until the overlap drops below the threshold.

The answer and answer_sentence are never modified: only the question text
changes, so the retrieval ground truth stays intact.
"""

from __future__ import annotations

from src.answer_matching import _STOPWORDS, _tokens, content_token_overlap
from src.config import LLMProviderConfig
from src.providers import chat_completion_json_async

DEFAULT_MAX_OVERLAP = 0.35
DEFAULT_MAX_RETRIES = 3


def _overlapping_content_tokens(question: str, answer_sentence: str) -> list[str]:
    """Content tokens shared by the question and the answer sentence."""
    sentence_tokens = set(_tokens(answer_sentence))
    seen: set[str] = set()
    shared = []
    for token in _tokens(question):
        if token in _STOPWORDS or token in seen:
            continue
        seen.add(token)
        if token in sentence_tokens:
            shared.append(token)
    return shared


def _build_prompt(
    question: str,
    answer: str,
    answer_sentence: str,
    forbidden: list[str],
) -> str:
    forbidden_line = (
        f"4. Do NOT use any of these words or their inflections: {', '.join(forbidden)}\n"
        if forbidden
        else ""
    )
    return f"""You rewrite benchmark questions to make keyword-based retrieval harder while keeping the question answerable.

Rewrite the question below so that:
1. It asks for exactly the same fact - the given answer must remain the single correct answer.
2. It shares as little vocabulary as possible with the source sentence: use synonyms, paraphrases, or ask from a different angle.
3. It does NOT contain the answer itself.
{forbidden_line}
Question: {question}
Answer: {answer}
Source sentence: {answer_sentence}

Return ONLY a JSON object: {{"question": "rewritten question"}}"""


async def paraphrase_question_async(
    question: str,
    answer: str,
    answer_sentence: str,
    config: LLMProviderConfig,
    *,
    max_overlap: float = DEFAULT_MAX_OVERLAP,
    max_retries: int = DEFAULT_MAX_RETRIES,
) -> dict:
    """
    Paraphrase one question away from its answer sentence's vocabulary.

    Returns a dict with:
        question: best paraphrase found (lowest overlap);
        overlap: its content-token overlap with the answer sentence;
        original_overlap: overlap of the original question;
        accepted: True when overlap <= max_overlap.

    Falls back to the original question only if every attempt fails to parse.
    """
    original_overlap = content_token_overlap(question, answer_sentence)
    best_question = question
    best_overlap = original_overlap
    forbidden = _overlapping_content_tokens(question, answer_sentence)

    for attempt in range(max_retries):
        try:
            parsed = await chat_completion_json_async(
                _build_prompt(question, answer, answer_sentence, forbidden),
                config=config,
                temperature=0.6 + 0.1 * attempt,
                timeout=60.0,
            )
        except Exception as exc:
            print(f"  [WARN] paraphrase attempt {attempt + 1} failed: {exc}")
            continue

        candidate = str(parsed.get("question", "")).strip()
        if not candidate:
            continue

        overlap = content_token_overlap(candidate, answer_sentence)
        if overlap < best_overlap or (best_question == question and candidate != question):
            best_question = candidate
            best_overlap = overlap
        if overlap <= max_overlap:
            break
        # Tighten the constraint with the words that still leak through.
        forbidden = sorted(
            set(forbidden) | set(_overlapping_content_tokens(candidate, answer_sentence))
        )

    return {
        "question": best_question,
        "overlap": best_overlap,
        "original_overlap": original_overlap,
        "accepted": best_overlap <= max_overlap,
    }
