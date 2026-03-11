"""
FastAPI 应用（可选 API 模式）。

启动: python -m research_engine serve --port 38417
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from . import __version__
from .config import load_config, settings
from .schemas import (
    HealthResponse,
    LLMConfigRequest,
    LLMConfigResponse,
    PipelineRunRequest,
    PipelineStatusResponse,
)
from .services.events import event_bus
from .services.pipeline import pipeline_service

app = FastAPI(
    title="Research Idea Discovery API",
    version=__version__,
    description="科研 Idea 发掘与研究计划生成系统 API",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 运行中的 session 跟踪
_running_sessions: dict[str, dict] = {}


@app.get("/health", response_model=HealthResponse)
def healthcheck() -> HealthResponse:
    return HealthResponse(status="ok", version=__version__)


@app.post("/api/pipeline/run", response_model=PipelineStatusResponse)
async def run_pipeline(request: PipelineRunRequest) -> PipelineStatusResponse:
    """启动流水线（异步执行）。"""
    config_path = Path(request.config_path)
    if not config_path.exists():
        raise HTTPException(status_code=400, detail=f"配置文件不存在: {config_path}")

    try:
        config = load_config(config_path)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"配置文件解析失败: {e}")

    if request.stages:
        config.pipeline.stages = request.stages
    if request.resume_from:
        config.pipeline.resume_from = request.resume_from
    if request.continue_from:
        config.pipeline.continue_from = request.continue_from

    session_id = str(uuid4())[:8]
    session_dir = Path(request.session_dir) if request.session_dir else None
    base_dir = config_path.parent

    _running_sessions[session_id] = {"status": "running", "stages_completed": []}

    async def _bg_run():
        try:
            results = await pipeline_service.run(
                config,
                session_dir=session_dir,
                session_id=session_id,
                base_dir=base_dir,
            )
            completed = [n for n, r in results.items() if r.success]
            _running_sessions[session_id] = {"status": "completed", "stages_completed": completed}
        except Exception as e:
            _running_sessions[session_id] = {"status": "failed", "error": str(e)}

    asyncio.create_task(_bg_run())

    return PipelineStatusResponse(
        session_id=session_id,
        status="running",
        message="流水线已启动。",
    )


@app.get("/api/pipeline/status/{session_id}", response_model=PipelineStatusResponse)
def get_pipeline_status(session_id: str) -> PipelineStatusResponse:
    info = _running_sessions.get(session_id)
    if not info:
        raise HTTPException(status_code=404, detail="未找到该 session。")
    return PipelineStatusResponse(
        session_id=session_id,
        status=info.get("status", "unknown"),
        stages_completed=info.get("stages_completed", []),
        message=info.get("error", ""),
    )


@app.get("/api/pipeline/events/{session_id}")
async def stream_events(session_id: str):
    """SSE 事件流（进度推送）。"""
    async def event_generator():
        async for event in event_bus.stream(session_id):
            yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.post("/api/llm/test", response_model=LLMConfigResponse)
async def test_llm(request: LLMConfigRequest) -> LLMConfigResponse:
    """测试 LLM 连通性。"""
    from .services.llm import LLMService
    from .config import LLMConfig

    llm_config = LLMConfig(
        provider=request.provider,
        base_url=request.base_url or "",
        api_key=request.api_key or "",
        model=request.model or "",
    )
    llm = LLMService(llm_config)

    if not llm.is_configured:
        return LLMConfigResponse(
            provider=request.provider,
            configured=False,
            message="LLM 配置不完整。",
        )

    try:
        result = await llm.chat([{"role": "user", "content": "Reply with OK."}])
        return LLMConfigResponse(
            provider=request.provider,
            configured=True,
            model=request.model,
            message=f"连接成功: {result[:50]}",
        )
    except Exception as e:
        return LLMConfigResponse(
            provider=request.provider,
            configured=False,
            model=request.model,
            message=f"连接失败: {e}",
        )
