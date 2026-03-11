"""
CLI 入口：python -m research_engine

用法:
    python -m research_engine run --config research_config.yaml
    python -m research_engine run --config research_config.yaml --stages 1,2
    python -m research_engine run --config research_config.yaml --resume-from 3
    python -m research_engine run --config research_config.yaml --continue-from 2026-03-10_14-30-00
    python -m research_engine run --config research_config.yaml --session output/2026-03-10_14-30-00
    python -m research_engine serve --port 38417
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path


def _setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def _run_pipeline(args: argparse.Namespace) -> None:
    """执行研究流水线。"""
    from .config import load_config
    from .services.pipeline import pipeline_service

    config_path = Path(args.config)
    config = load_config(config_path)

    if args.stages:
        config.pipeline.stages = [int(s.strip()) for s in args.stages.split(",")]
    if args.resume_from:
        config.pipeline.resume_from = args.resume_from
    if args.continue_from:
        config.pipeline.continue_from = args.continue_from

    session_dir = Path(args.session) if args.session else None
    base_dir = config_path.parent

    async def _async_run():
        results = await pipeline_service.run(
            config,
            session_dir=session_dir,
            base_dir=base_dir,
        )

        print("\n" + "=" * 60)
        print("流水线执行完毕")
        print("=" * 60)
        for stage_num in sorted(results):
            r = results[stage_num]
            status_icon = "+" if r.success else "x"
            degraded = " [降级]" if r.metadata.degraded else ""
            print(f"  Stage {stage_num} ({r.stage_name}): {status_icon} {r.metadata.status}{degraded}"
                  f" ({r.metadata.duration_ms}ms)")
        print("=" * 60)

    asyncio.run(_async_run())


def _serve(args: argparse.Namespace) -> None:
    """启动 FastAPI 服务。"""
    import uvicorn
    from .main import app

    uvicorn.run(app, host=args.host, port=args.port)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="research_engine",
        description="科研 Idea 发掘与研究计划生成系统",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="详细日志")

    subparsers = parser.add_subparsers(dest="command")

    # run 命令
    run_parser = subparsers.add_parser("run", help="执行研究流水线")
    run_parser.add_argument("--config", "-c", required=True, help="YAML 配置文件路径")
    run_parser.add_argument("--stages", "-s", default=None,
                            help="要运行的阶段，逗号分隔 (如 1,2,3)")
    run_parser.add_argument("--resume-from", "-r", type=int, default=None,
                            help="从指定阶段继续")
    run_parser.add_argument("--continue-from", default=None,
                            help="沿指定 session 继续执行（可填 session 名或路径）")
    run_parser.add_argument("--session", default=None, help="已有 session 目录路径")

    # serve 命令
    serve_parser = subparsers.add_parser("serve", help="启动 API 服务")
    serve_parser.add_argument("--host", default="127.0.0.1")
    serve_parser.add_argument("--port", type=int, default=38417)

    args = parser.parse_args()
    _setup_logging(args.verbose)

    if args.command == "serve":
        _serve(args)
    elif args.command == "run":
        _run_pipeline(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
