"""Validate embedding and LLM provider configuration.

By default this command performs a dry-run config check without network calls.
Use `--run` for a minimal embedding/LLM request.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import asdict
from typing import Any

from dotenv import load_dotenv

from src.providers import (
    build_embedding_model,
    chat_completion_json,
    embedding_config_from_env,
    llm_config_from_env,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check provider configuration and connectivity")
    parser.add_argument("--run", action="store_true", help="Perform minimal provider calls")
    parser.add_argument(
        "--require-api-keys",
        action="store_true",
        help="Fail dry-run if a configured API-key environment variable is unset",
    )
    parser.add_argument("--skip-embedding", action="store_true")
    parser.add_argument("--skip-llm", action="store_true")
    parser.add_argument("--timeout", type=float, default=30.0)

    parser.add_argument("--embedding-provider")
    parser.add_argument("--embedding-model")
    parser.add_argument("--embedding-api-key-env")
    parser.add_argument("--embedding-base-url")
    parser.add_argument("--embedding-text", default="Provider smoke test text.")

    parser.add_argument("--llm-provider")
    parser.add_argument("--llm-model")
    parser.add_argument("--llm-api-key-env")
    parser.add_argument("--llm-base-url")
    parser.add_argument(
        "--llm-prompt",
        default='Return exactly this JSON object: {"ok": true, "provider_smoke": true}',
    )
    return parser.parse_args()


def _redacted_config(config: Any) -> dict[str, Any]:
    data = asdict(config)
    if data.get("api_key"):
        data["api_key"] = "***"
    return data


def _check_key(config: Any, require_api_keys: bool) -> None:
    if not config.api_key_env:
        return
    if os.getenv(config.api_key_env):
        print(f"[OK] {config.api_key_env} is set")
        return
    message = f"[WARN] {config.api_key_env} is not set"
    if require_api_keys:
        raise ValueError(message.replace("[WARN]", "[ERROR]"))
    print(message)


def _run_embedding_check(args: argparse.Namespace) -> None:
    config = embedding_config_from_env(
        provider=args.embedding_provider,
        model=args.embedding_model,
        api_key_env=args.embedding_api_key_env,
        base_url=args.embedding_base_url,
    )
    print(f"[INFO] Embedding config: {_redacted_config(config)}")
    _check_key(config, args.require_api_keys)
    if not args.run:
        return

    model = build_embedding_model(config)
    embedding = model.get_text_embedding(args.embedding_text)
    print(f"[OK] Embedding request returned {len(embedding)} dimensions")


def _run_llm_check(args: argparse.Namespace) -> None:
    config = llm_config_from_env(
        provider=args.llm_provider,
        model=args.llm_model,
        api_key_env=args.llm_api_key_env,
        base_url=args.llm_base_url,
    )
    print(f"[INFO] LLM config: {_redacted_config(config)}")
    _check_key(config, args.require_api_keys)
    if not args.run:
        return

    payload = chat_completion_json(args.llm_prompt, config=config, temperature=0.0, timeout=args.timeout)
    print(f"[OK] LLM request returned JSON keys: {', '.join(sorted(payload))}")


def main() -> int:
    load_dotenv()
    args = parse_args()
    try:
        if not args.skip_embedding:
            _run_embedding_check(args)
        if not args.skip_llm:
            _run_llm_check(args)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"[ERROR] Provider check failed: {exc}", file=sys.stderr)
        return 1

    if not args.run:
        print("[OK] Provider dry-run completed. Use --run to perform live calls.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
