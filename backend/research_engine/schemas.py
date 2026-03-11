"""
API 请求/响应 schema（Pydantic v2 模型）。
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class PipelineRunRequest(BaseModel):
    """启动流水线的请求。"""
    config_path: str = Field(..., description="YAML 配置文件路径")
    stages: list[int] = Field(default_factory=lambda: [1, 2, 3, 4, 5])
    resume_from: int | None = None
    continue_from: str | None = None
    session_dir: str | None = None


class PipelineStatusResponse(BaseModel):
    session_id: str
    status: str
    stages_completed: list[int] = Field(default_factory=list)
    current_stage: int | None = None
    session_dir: str = ""
    message: str = ""


class LLMConfigRequest(BaseModel):
    provider: str = "openai_compatible"
    base_url: str | None = None
    api_key: str | None = None
    model: str | None = None


class LLMConfigResponse(BaseModel):
    provider: str
    configured: bool
    model: str | None = None
    message: str = ""


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str = ""
