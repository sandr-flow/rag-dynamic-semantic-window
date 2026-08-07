"""FinanceBench dataset loader for benchmark testing.

Loads the open-source subset of FinanceBench (Patronus AI): 150 questions
over SEC filings (10-K/10-Q/8-K/earnings) of US public companies, each with
a human-annotated evidence string from the source document.

Questions and document metadata come from the GitHub repo's JSONL files;
the filing PDFs are downloaded per document and extracted with ``pypdf``.
Everything is cached under ``data/financebench_cache/`` so repeated
prepare runs are offline.

Output items follow the stand dataset schema: one item per document with
``qa_pairs`` whose ``answer_sentence`` is the first evidence sentence that
actually resolves in the extracted PDF text (extraction can mangle table
regions, so each evidence is verified before use).
"""

from __future__ import annotations

import json
from pathlib import Path

import httpx

from src.answer_matching import find_answer_sentence_idx, normalize_answer_text
from src.utils import split_into_sentences

FINANCEBENCH_RAW = "https://raw.githubusercontent.com/patronus-ai/financebench/main"
QUESTIONS_FILE = "data/financebench_open_source.jsonl"
DEFAULT_CACHE_DIR = Path("data") / "financebench_cache"

# Evidence fragments shorter than this are things like lone table cells
# ("Total revenue") that match all over a filing; skip them as anchors.
_MIN_EVIDENCE_SENTENCE_CHARS = 40


def _download(relative_path: str, cache_dir: Path, *, binary: bool = False) -> Path:
    """Fetch a repo file into the cache (no-op when already present)."""
    target = cache_dir / relative_path.replace("/", "_")
    if target.exists() and target.stat().st_size > 0:
        return target
    cache_dir.mkdir(parents=True, exist_ok=True)
    url = f"{FINANCEBENCH_RAW}/{relative_path}"
    with httpx.Client(follow_redirects=True, timeout=120.0) as client:
        response = client.get(url)
        response.raise_for_status()
        target.write_bytes(response.content if binary else response.text.encode("utf-8"))
    return target


def load_financebench_questions(cache_dir: Path = DEFAULT_CACHE_DIR) -> list[dict]:
    """Load the 150 open-source FinanceBench question records."""
    path = _download(QUESTIONS_FILE, cache_dir)
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def extract_pdf_text(pdf_path: Path) -> str:
    """Extract plain text from a filing PDF, page by page."""
    from pypdf import PdfReader

    reader = PdfReader(str(pdf_path))
    pages = []
    for page in reader.pages:
        text = page.extract_text() or ""
        if text.strip():
            pages.append(text)
    return "\n".join(pages)


def _evidence_texts(record: dict) -> list[str]:
    """Pull evidence strings out of a question record (schema-tolerant)."""
    evidence = record.get("evidence") or []
    if isinstance(evidence, str):
        return [evidence]
    texts = []
    for entry in evidence:
        if isinstance(entry, str):
            texts.append(entry)
        elif isinstance(entry, dict):
            text = entry.get("evidence_text") or entry.get("evidence_text_full_page") or ""
            if text:
                texts.append(text)
    return texts


def resolve_answer_sentence(evidence_texts: list[str], doc_text_norm: str) -> str:
    """Pick the evidence sentence to use as the extractive ground truth.

    Evidence strings are often multi-sentence or table fragments; the PDF
    extraction may not reproduce them verbatim. Prefer the first evidence
    sentence long enough to be distinctive that appears (normalized) in the
    document text; fall back to the longest evidence sentence so
    ``find_answer_sentence_idx``'s fuzzy path still has something to anchor.
    """
    fallback = ""
    for evidence in evidence_texts:
        for sentence in split_into_sentences(evidence):
            sentence = sentence.strip()
            if len(sentence) < _MIN_EVIDENCE_SENTENCE_CHARS:
                continue
            if len(sentence) > len(fallback):
                fallback = sentence
            if normalize_answer_text(sentence) in doc_text_norm:
                return sentence
    if fallback:
        return fallback
    longest = max((e.strip() for e in evidence_texts), key=len, default="")
    return longest


def load_financebench_items(
    cache_dir: Path = DEFAULT_CACHE_DIR,
    *,
    max_docs: int | None = None,
) -> list[dict]:
    """Build stand dataset items (one per filing) from the open subset.

    Returns items shaped like the other loaders' output:
    ``{"id", "title", "text", "qa_pairs": [{"question", "answer",
    "answer_sentence", "financebench_id"}]}``. Prints per-document answer
    resolution stats: questions whose answer sentence cannot be located in
    the extracted text are kept (they simply score ``answer_sentence_idx=-1``
    in HPO corpora), so the printed match rate is the number to watch.
    """
    records = load_financebench_questions(cache_dir)

    by_doc: dict[str, list[dict]] = {}
    for record in records:
        by_doc.setdefault(record["doc_name"], []).append(record)

    doc_names = sorted(by_doc)
    if max_docs is not None:
        doc_names = doc_names[:max_docs]

    items: list[dict] = []
    matched = 0
    total = 0
    for doc_idx, doc_name in enumerate(doc_names):
        try:
            pdf_path = _download(f"pdfs/{doc_name}.pdf", cache_dir, binary=True)
            text = extract_pdf_text(pdf_path)
        except Exception as exc:  # noqa: BLE001 - one bad filing must not kill the prep
            print(f"  [WARN] Failed to load {doc_name}: {exc}")
            continue
        if not text.strip():
            print(f"  [WARN] Empty extraction, skipping: {doc_name}")
            continue
        doc_text_norm = normalize_answer_text(text)
        doc_sentences = split_into_sentences(text)

        qa_pairs = []
        doc_matched = 0
        for record in by_doc[doc_name]:
            evidence_texts = _evidence_texts(record)
            answer_sentence = resolve_answer_sentence(evidence_texts, doc_text_norm)
            if not answer_sentence:
                print(f"  [WARN] No evidence text: {record.get('financebench_id')}")
                continue
            if find_answer_sentence_idx(doc_sentences, answer_sentence) >= 0:
                doc_matched += 1
            qa_pairs.append(
                {
                    "question": record["question"],
                    "answer": record.get("answer", ""),
                    "answer_sentence": answer_sentence,
                    "financebench_id": record.get("financebench_id", ""),
                }
            )

        matched += doc_matched
        total += len(qa_pairs)
        items.append(
            {
                "id": doc_idx,
                "title": doc_name,
                "text": text,
                "qa_pairs": qa_pairs,
            }
        )
        print(
            f"  [INFO] [{doc_idx + 1}/{len(doc_names)}] {doc_name}: "
            f"{len(text)} chars, {doc_matched}/{len(qa_pairs)} answers resolved"
        )

    if total:
        print(f"[INFO] FinanceBench answer resolution: {matched}/{total} ({matched / total:.0%})")
    return items
