"""流水线集成测试。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from research_engine.config import (
    LLMConfig,
    PaperSearchConfig,
    PipelineConfig,
    load_config,
)
from research_engine.models import StageMetadata, StageResult
from research_engine.services.output import output_service
from research_engine.services.pipeline import pipeline_service
from research_engine.services.retry import get_retry_wait_seconds
from research_engine.stages.base import StageBase, StageContext


class TestOutputService:
    def test_create_session_dir(self, tmp_path):
        session_dir = output_service.create_session_dir(tmp_path)
        assert session_dir.is_dir()
        assert session_dir.parent == tmp_path

    def test_find_latest_session(self, tmp_path):
        d1 = tmp_path / "2025-01-01_00-00-00"
        d1.mkdir()
        d2 = tmp_path / "2025-06-01_12-00-00"
        d2.mkdir()
        latest = output_service.find_latest_session(tmp_path)
        assert latest == d2

    def test_save_session_config(self, tmp_path):
        output_service.save_session_config(tmp_path, {"test": True})
        path = tmp_path / "config_snapshot.json"
        assert path.exists()
        data = json.loads(path.read_text())
        assert data["test"] is True

    def test_resolve_session_dir(self, tmp_path):
        output_base = tmp_path / "output"
        session_dir = output_base / "resume_case"
        session_dir.mkdir(parents=True)

        resolved = output_service.resolve_session_dir(output_base, "resume_case", base_dir=tmp_path)
        assert resolved == session_dir


class TestRetryHelper:
    def test_exponential_backoff(self):
        assert get_retry_wait_seconds(1) == 10
        assert get_retry_wait_seconds(2) == 20
        assert get_retry_wait_seconds(3) == 40


class TestConfigLoading:
    def test_load_example_config(self):
        example = Path(__file__).resolve().parents[2] / "research_config.example.yaml"
        if not example.exists():
            pytest.skip("示例配置文件不存在")
        config = load_config(example)
        assert config.research_direction.name
        assert len(config.pipeline.stages) == 5
        assert isinstance(config.pipeline, PipelineConfig)
        assert isinstance(config.paper_search, PaperSearchConfig)
        assert isinstance(config.llm, LLMConfig)
        assert config.pipeline.output_dir
        assert config.llm.model
        assert config.paper_search.sources
        assert config.paper_search.request_max_retries == 3

    def test_load_nonexistent_raises(self):
        with pytest.raises(FileNotFoundError):
            load_config("nonexistent.yaml")


class FakeStage1(StageBase):
    stage_number = 1
    stage_name = "fake_stage_1"
    run_count = 0

    async def run(self, ctx: StageContext) -> StageResult:
        type(self).run_count += 1
        return StageResult(
            stage_number=self.stage_number,
            stage_name=self.stage_name,
            success=True,
            data={"stage": 1, "source": "fresh"},
            metadata=StageMetadata(stage_number=self.stage_number, stage_name=self.stage_name),
            markdown="# fake stage 1",
        )


class FakeStage2(StageBase):
    stage_number = 2
    stage_name = "fake_stage_2"
    run_count = 0
    previous_stage_1: dict | None = None

    async def run(self, ctx: StageContext) -> StageResult:
        type(self).run_count += 1
        type(self).previous_stage_1 = ctx.get_previous(1)
        return StageResult(
            stage_number=self.stage_number,
            stage_name=self.stage_name,
            success=True,
            data={"stage": 2, "previous_stage_1": ctx.get_previous(1)},
            metadata=StageMetadata(stage_number=self.stage_number, stage_name=self.stage_name),
            markdown="# fake stage 2",
        )


def _write_stage_snapshot(
    session_dir: Path,
    *,
    stage_number: int,
    stage_name: str,
    data: dict,
    status: str = "completed",
) -> None:
    stage_dir = session_dir / f"stage_{stage_number}"
    stage_dir.mkdir(parents=True, exist_ok=True)
    (stage_dir / f"{stage_name}.json").write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (stage_dir / "metadata.json").write_text(
        json.dumps(
            {
                "stage_number": stage_number,
                "stage_name": stage_name,
                "status": status,
                "started_at": None,
                "finished_at": None,
                "duration_ms": 0,
                "error": None,
                "degraded": False,
                "degraded_reason": None,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


class TestPipelineResume:
    @pytest.fixture(autouse=True)
    def _reset_fake_stages(self):
        FakeStage1.run_count = 0
        FakeStage2.run_count = 0
        FakeStage2.previous_stage_1 = None

    @pytest.mark.asyncio
    async def test_continue_from_runs_from_first_unfinished_stage(self, sample_config, tmp_path, monkeypatch):
        output_base = Path(sample_config.pipeline.output_dir)
        session_dir = output_base / "resume_case"
        session_dir.mkdir(parents=True)

        _write_stage_snapshot(
            session_dir,
            stage_number=1,
            stage_name="fake_stage_1",
            data={"stage": 1, "source": "cached"},
            status="completed",
        )
        _write_stage_snapshot(
            session_dir,
            stage_number=2,
            stage_name="fake_stage_2",
            data={"stage": 2, "source": "stale"},
            status="degraded",
        )

        sample_config.pipeline.stages = [1, 2]
        sample_config.pipeline.continue_from = "resume_case"

        monkeypatch.setattr(
            "research_engine.services.pipeline.STAGE_REGISTRY",
            {1: FakeStage1, 2: FakeStage2},
        )

        results = await pipeline_service.run(sample_config, base_dir=tmp_path)

        assert FakeStage1.run_count == 0
        assert FakeStage2.run_count == 1
        assert FakeStage2.previous_stage_1 == {"stage": 1, "source": "cached"}
        assert results[1].metadata.status == "completed"
        assert results[2].metadata.status == "completed"

        summary = json.loads((session_dir / "summary.json").read_text(encoding="utf-8"))
        assert summary["status"] == "completed"
        assert summary["stages"] == {"1": "completed", "2": "completed"}

    @pytest.mark.asyncio
    async def test_resume_from_forces_rerun_of_target_stage(self, sample_config, tmp_path, monkeypatch):
        output_base = Path(sample_config.pipeline.output_dir)
        session_dir = output_base / "manual_resume_case"
        session_dir.mkdir(parents=True)

        _write_stage_snapshot(
            session_dir,
            stage_number=1,
            stage_name="fake_stage_1",
            data={"stage": 1, "source": "cached"},
            status="completed",
        )
        _write_stage_snapshot(
            session_dir,
            stage_number=2,
            stage_name="fake_stage_2",
            data={"stage": 2, "source": "old"},
            status="completed",
        )

        sample_config.pipeline.stages = [1, 2]
        sample_config.pipeline.resume_from = 2

        monkeypatch.setattr(
            "research_engine.services.pipeline.STAGE_REGISTRY",
            {1: FakeStage1, 2: FakeStage2},
        )

        results = await pipeline_service.run(
            sample_config,
            session_dir=session_dir,
            base_dir=tmp_path,
        )

        assert FakeStage1.run_count == 0
        assert FakeStage2.run_count == 1
        assert FakeStage2.previous_stage_1 == {"stage": 1, "source": "cached"}
        assert results[1].metadata.status == "completed"
        assert results[2].data["previous_stage_1"] == {"stage": 1, "source": "cached"}
