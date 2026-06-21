from __future__ import annotations

import argparse
import json

from .config import DagParams, OpenAIConfig
from .core import build_dag_batch
from .io import read_jsonl


def _add_model_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--n", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=None)


def _openai_config(args: argparse.Namespace) -> OpenAIConfig:
    overrides = {
        "api_key": args.api_key,
        "base_url": args.base_url,
        "model": args.model,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
        "n": args.n,
        "timeout": args.timeout,
    }
    clean = {key: value for key, value in overrides.items() if value is not None}
    return OpenAIConfig.from_env(**clean)


def _dag_params(args: argparse.Namespace) -> DagParams:
    return DagParams(
        regen_limit=args.regen_limit,
        main_path_cap=args.main_path_cap,
        other_leaf_cap=args.other_leaf_cap,
        random_seed=args.random_seed,
    )


def cmd_build(args: argparse.Namespace) -> None:
    rows = read_jsonl(args.input)
    result = build_dag_batch(
        rows,
        openai_config=_openai_config(args),
        dag_params=_dag_params(args),
        num_threads=args.num_threads,
        output_path=args.output,
        resume=args.resume,
        keywords=tuple(args.keywords),
    )
    if not args.output:
        print(json.dumps(result, ensure_ascii=False, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="trm-dag")
    sub = parser.add_subparsers(dest="command", required=True)

    build = sub.add_parser("build")
    build.add_argument("--input", required=True)
    build.add_argument("--output", default=None)
    build.add_argument("--num-threads", type=int, default=None)
    build.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    build.add_argument("--rebuild", dest="resume", action="store_false")
    build.add_argument("--keywords", nargs="*", default=["Step", "Then", "Next", "Finally"])
    build.add_argument("--regen-limit", type=int, default=5)
    build.add_argument("--main-path-cap", type=int, default=8)
    build.add_argument("--other-leaf-cap", type=int, default=5)
    build.add_argument("--random-seed", type=int, default=None)
    _add_model_args(build)
    build.set_defaults(func=cmd_build)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)
