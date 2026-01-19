"""LLM-based question generator for benchmark testing."""

import json
import os
from typing import Optional

import httpx
from dotenv import load_dotenv

load_dotenv()


def generate_questions(
    text: str,
    num_questions: int = 10,
    api_key: Optional[str] = None,
) -> list[str]:
    """
    Generate benchmark questions using Mistral API with JSON mode.

    Sends article text once, receives JSON array of questions.

    Args:
        text: Source text to generate questions about.
        num_questions: Number of questions to generate.
        api_key: Mistral API key (defaults to MISTRAL_API_KEY env var).

    Returns:
        List of generated questions.

    Raises:
        ValueError: If API key not provided.
        httpx.HTTPError: If API request fails.
    """
    api_key = api_key or os.getenv("MISTRAL_API_KEY")
    if not api_key:
        raise ValueError("MISTRAL_API_KEY not set")

    # Truncate text if too long (Mistral context limit)
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

    response = httpx.post(
        "https://api.mistral.ai/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": "mistral-small-latest",
            "messages": [{"role": "user", "content": prompt}],
            "response_format": {"type": "json_object"},
            "temperature": 0.7,
        },
        timeout=60.0,
    )
    response.raise_for_status()

    result = response.json()
    content = result["choices"][0]["message"]["content"]
    parsed = json.loads(content)

    return parsed.get("questions", [])[:num_questions]


def generate_qa_pairs(
    text: str,
    num_questions: int = 10,
    api_key: Optional[str] = None,
) -> list[dict]:
    """
    Generate question-answer pairs using Mistral API with JSON mode.
    Includes validation to ensure answers are extractive.
    """
    api_key = api_key or os.getenv("MISTRAL_API_KEY")
    if not api_key:
        raise ValueError("MISTRAL_API_KEY not set")

    # Truncate text if too long (Mistral context limit)
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
            print(f"  🔄 Retrying QA generation (attempt {retry_count + 1}, found {len(valid_pairs)}/{num_questions} valid)...")

        try:
            response = httpx.post(
                "https://api.mistral.ai/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": "mistral-small-latest",
                    "messages": [{"role": "user", "content": prompt}],
                    "response_format": {"type": "json_object"},
                    "temperature": 0.5 + (0.1 * retry_count),  # Slightly increase temperature on retry
                },
                timeout=90.0,
            )
            response.raise_for_status()

            result = response.json()
            content = result["choices"][0]["message"]["content"]
            parsed = json.loads(content)
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
            print(f"  ⚠️ Error during QA generation retry: {e}")
        
        retry_count += 1

    return valid_pairs[:num_questions]


async def generate_qa_pairs_async(
    text: str,
    num_questions: int = 10,
    api_key: Optional[str] = None,
) -> list[dict]:
    """
    Async version of generate_qa_pairs for parallel batch processing.
    """
    api_key = api_key or os.getenv("MISTRAL_API_KEY")
    if not api_key:
        raise ValueError("MISTRAL_API_KEY not set")

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

    async with httpx.AsyncClient(timeout=90.0) as client:
        while len(valid_pairs) < num_questions and retry_count < max_retries:
            try:
                response = await client.post(
                    "https://api.mistral.ai/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": "mistral-small-latest",
                        "messages": [{"role": "user", "content": prompt}],
                        "response_format": {"type": "json_object"},
                        "temperature": 0.5 + (0.1 * retry_count),
                    },
                )
                response.raise_for_status()

                result = response.json()
                content = result["choices"][0]["message"]["content"]
                parsed = json.loads(content)
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
