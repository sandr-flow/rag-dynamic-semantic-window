"""LLM-based question generator for benchmark testing."""

import os

from dotenv import load_dotenv

from src.config import LLMProviderConfig
from src.providers import (
    chat_completion_json,
    chat_completion_json_async,
    llm_config_from_env,
)

load_dotenv()


def _provider_config(api_key: str | None = None) -> LLMProviderConfig:
    config = llm_config_from_env()
    if api_key:
        config.api_key = api_key
        if not config.api_key_env:
            config.api_key_env = "MISTRAL_API_KEY"
    return config


def generate_questions(
    text: str,
    num_questions: int = 10,
    api_key: str | None = None,
) -> list[str]:
    """
    Generate benchmark questions using the configured OpenAI-compatible chat provider.

    Sends article text once, receives JSON array of questions.

    Args:
        text: Source text to generate questions about.
        num_questions: Number of questions to generate.
        api_key: Optional direct API key override.

    Returns:
        List of generated questions.

    Raises:
        ValueError: If API key not provided.
        httpx.HTTPError: If API request fails.
    """
    config = _provider_config(api_key)
    if not config.api_key and config.api_key_env and not os.getenv(config.api_key_env):
        raise ValueError(f"{config.api_key_env} not set")

    # Keep prompts bounded for hosted models with moderate context windows.
    max_chars = 30000
    if len(text) > max_chars:
        text = text[:max_chars] + "..."

    prompt = f"""You are a benchmark question generator. Read the following article and generate {num_questions} diverse, specific questions that can be answered using the article content.

Requirements:
- Questions should be factual and have clear answers in the text
- Mix of "what", "how", "why", "when", "who" questions
- Avoid yes/no questions
- Questions should require understanding, not just keyword matching

Article:
{text}

Return ONLY a JSON object with this exact format:
{{"questions": ["question 1", "question 2", ...]}}"""

    parsed = chat_completion_json(prompt, config=config, temperature=0.7, timeout=60.0)
    return parsed.get("questions", [])[:num_questions]


def generate_qa_pairs(
    text: str,
    num_questions: int = 10,
    api_key: str | None = None,
) -> list[dict]:
    """
    Generate question-answer pairs using the configured OpenAI-compatible chat provider.
    Includes validation to ensure answers are extractive.
    """
    config = _provider_config(api_key)
    if not config.api_key and config.api_key_env and not os.getenv(config.api_key_env):
        raise ValueError(f"{config.api_key_env} not set")

    # Keep prompts bounded for hosted models with moderate context windows.
    max_chars = 30000
    if len(text) > max_chars:
        text = text[:max_chars] + "..."

    prompt = f"""You are a benchmark QA generator. Read the article and generate {num_questions} question-answer pairs.

CRITICAL REQUIREMENTS:
1. Each answer MUST be a short, factual answer (a word, name, date, number, or short phrase)
2. Each answer_sentence MUST be an EXACT sentence copied verbatim from the article that contains the answer
3. Do NOT paraphrase or modify the answer_sentence - copy it exactly as it appears
4. Avoid yes/no questions
5. Questions should be diverse (who, what, when, where, why, how)

Article:
{text}

Return ONLY a JSON object with this exact format:
{{"qa_pairs": [
  {{"question": "What is X?", "answer": "Y", "answer_sentence": "The exact sentence from article containing Y."}},
  ...
]}}"""

    valid_pairs = []
    max_retries = 3
    retry_count = 0

    while len(valid_pairs) < num_questions and retry_count < max_retries:
        if retry_count > 0:
            print(
                "  [INFO] Retrying QA generation "
                f"(attempt {retry_count + 1}, found {len(valid_pairs)}/{num_questions} valid)..."
            )

        try:
            parsed = chat_completion_json(
                prompt,
                config=config,
                temperature=0.5 + (0.1 * retry_count),
                timeout=90.0,
            )
            batch = parsed.get("qa_pairs", [])

            for pair in batch:
                if len(valid_pairs) >= num_questions:
                    break
                
                # Validation logic
                q = pair.get("question", "")
                a = pair.get("answer", "")
                ans_sent = pair.get("answer_sentence", "")

                if not q or not a or not ans_sent:
                    continue

                # 1. Check if answer_sentence exists in original text (exact match)
                if ans_sent not in text:
                    # Try with minimal cleaning (some LLMs add quotes or trailing spaces)
                    clean_ans = ans_sent.strip(' \t\n\r"\'')
                    if clean_ans not in text:
                        continue
                    ans_sent = clean_ans

                # 2. Check if answer exists in answer_sentence
                if a.lower() not in ans_sent.lower():
                    continue
                
                # 3. Deduplicate
                if any(v["question"] == q for v in valid_pairs):
                    continue

                valid_pairs.append({
                    "question": q,
                    "answer": a,
                    "answer_sentence": ans_sent
                })

        except Exception as e:
            print(f"  [WARN] Error during QA generation retry: {e}")
        
        retry_count += 1

    return valid_pairs[:num_questions]


async def generate_qa_pairs_async(
    text: str,
    num_questions: int = 10,
    api_key: str | None = None,
) -> list[dict]:
    """
    Async version of generate_qa_pairs for parallel batch processing.
    """
    config = _provider_config(api_key)
    if not config.api_key and config.api_key_env and not os.getenv(config.api_key_env):
        raise ValueError(f"{config.api_key_env} not set")

    max_chars = 30000
    if len(text) > max_chars:
        text = text[:max_chars] + "..."

    prompt = f"""You are a benchmark QA generator. Read the article and generate {num_questions} question-answer pairs.

CRITICAL REQUIREMENTS:
1. Each answer MUST be a short, factual answer (a word, name, date, number, or short phrase)
2. Each answer_sentence MUST be an EXACT sentence copied verbatim from the article that contains the answer
3. Do NOT paraphrase or modify the answer_sentence - copy it exactly as it appears
4. Avoid yes/no questions
5. Questions should be diverse (who, what, when, where, why, how)

Article:
{text}

Return ONLY a JSON object with this exact format:
{{"qa_pairs": [
  {{"question": "What is X?", "answer": "Y", "answer_sentence": "The exact sentence from article containing Y."}},
  ...
]}}"""

    valid_pairs = []
    max_retries = 3
    retry_count = 0

    while len(valid_pairs) < num_questions and retry_count < max_retries:
        try:
            parsed = await chat_completion_json_async(
                prompt,
                config=config,
                temperature=0.5 + (0.1 * retry_count),
                timeout=90.0,
            )
            batch = parsed.get("qa_pairs", [])

            for pair in batch:
                if len(valid_pairs) >= num_questions:
                    break
                
                q = pair.get("question", "")
                a = pair.get("answer", "")
                ans_sent = pair.get("answer_sentence", "")

                if not q or not a or not ans_sent:
                    continue

                if ans_sent not in text:
                    clean_ans = ans_sent.strip(' \t\n\r"\'')
                    if clean_ans not in text:
                        continue
                    ans_sent = clean_ans

                if a.lower() not in ans_sent.lower():
                    continue
                
                if any(v["question"] == q for v in valid_pairs):
                    continue

                valid_pairs.append({
                    "question": q,
                    "answer": a,
                    "answer_sentence": ans_sent
                })

        except Exception:
            pass  # Squelch errors in async flow to allow retries
        
        retry_count += 1

    return valid_pairs[:num_questions]
