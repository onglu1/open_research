# Research Idea Discovery & Plan Generation

配置驱动的五阶段科研决策流水线，将一个种子研究方向自动加工成多个高质量候选研究课题，并给出排序结果和完整研究计划。

## 系统定位

这是一个**科研决策支持与选题生成**系统，不是论文搜索引擎，也不是自动写论文工具。

## 五阶段流水线

```
Stage 1: Direction Expansion    → 方向发散（核心 + 平行 + 交叉方向）
Stage 2: Frontier Scan          → 前沿扫描（arXiv / Semantic Scholar 论文抓取 + 态势分析）
Stage 3: Idea Discovery         → Idea 发掘（基于研究空白和趋势信号生成候选 Idea）
Stage 4: Feasibility Ranking    → 可行性排序（多维度量化评估 + Tier 分级）
Stage 5: Deep Analysis          → 深度分析（生成接近 Research Proposal 的完整报告）
```

每个阶段产出：
- **JSON** — 机器可读的结构化中间结果
- **Markdown** — 人类可读的报告

## 核心特性

- **配置驱动**：单个 YAML 文件定义研究方向、搜索参数、评估维度、资源约束和 LLM 设置
- **断点续跑**：支持从任意阶段继续，自动复用历史输出
- **知识库增强**：加载已有研究/Idea/论文笔记，参与去重、排序和深度分析
- **多 LLM 支持**：OpenAI 兼容 API，支持第三方 base_url 和自定义模型（如 zhipu、deepseek 等）
- **优雅降级**：API 失败 / JSON 解析失败 / LLM 不可用时均有 fallback 策略
- **多源检索**：arXiv API + Semantic Scholar API
- **结果归档**：按时间戳归档，便于审计和复现

## 快速开始

### 前置要求

- Python 3.11+
- 一个 OpenAI 兼容的 LLM API（如智谱 GLM、DeepSeek、OpenAI 等）

### 安装

```bash
# 从项目根目录安装
cd backend
python -m pip install -e ".[dev]"
cd ..
```

### 配置

```bash
# 复制示例配置
cp research_config.yaml.example research_config.yaml

# 编辑 research_config.yaml：
#   1. 填写你的研究方向
#   2. 配置 LLM API 信息（base_url, api_key, model）
```

LLM API Key 也可以通过环境变量设置（会自动替换配置中的 `${RESEARCH_LLM_API_KEY}`）：

```bash
export RESEARCH_LLM_API_KEY=your-api-key
```

### 运行

> 注意：所有命令均在**项目根目录**下运行（即 `research_config.yaml` 所在目录）。

```bash
# 完整五阶段流水线
python -m research_engine run --config research_config.yaml

# 只运行前两个阶段
python -m research_engine run --config research_config.yaml --stages 1,2

# 从 Stage 3 继续（复用 Stage 1/2 历史输出）
python -m research_engine run --config research_config.yaml --resume-from 3

# 沿某次历史调研继续（自动从第一个未完成 stage 开始）
python -m research_engine run --config research_config.yaml --continue-from 2026-03-10_14-30-00

# 指定已有 session 目录继续
python -m research_engine run --config research_config.yaml --session output/2026-03-10_14-30-00

# 查看详细日志
python -m research_engine -v run --config research_config.yaml
```

### API 模式（可选）

```bash
python -m research_engine serve --port 38417
```

端点：
- `POST /api/pipeline/run` — 启动流水线
- `GET /api/pipeline/status/{session_id}` — 查询状态
- `GET /api/pipeline/events/{session_id}` — SSE 事件流
- `POST /api/llm/test` — 测试 LLM 连通性
- `GET /health` — 健康检查

## 输出结构

```
output/
└── 2026-03-10_14-30-00/           # Session 目录（时间戳命名）
    ├── config_snapshot.json        # 本次运行的配置快照
    ├── summary.json / summary.md   # 运行摘要（每个 stage 结束后都会更新）
    ├── stage_1/
    │   ├── direction_expansion.json
    │   ├── direction_expansion_report.md
    │   └── metadata.json
    ├── stage_2/
    │   ├── frontier_scan.json
    │   ├── frontier_scan_report.md
    │   └── metadata.json
    ├── stage_3/ ...
    ├── stage_4/ ...
    └── stage_5/ ...
```

## 知识库

项目已包含 `knowledge/` 目录及模板文件，按需编辑即可：

```
knowledge/
├── existing_research.md    # 已有研究方向描述（Stage 1 参考）
├── existing_ideas.md       # 已有 Idea（Stage 3 去重）
└── paper_notes/            # 已读论文笔记（Stage 4/5 补充上下文）
    ├── paper1.md
    └── paper2.md
```

## 测试

```bash
cd backend
pytest

# 带覆盖率报告
pytest --cov=research_engine
```

## 项目结构

```
open_research/
├── README.md
├── research_config.yaml.example    # 示例配置（复制后编辑）
├── research_config.yaml            # 实际配置（gitignored）
├── knowledge/                      # 个人知识库
│   ├── existing_research.md
│   ├── existing_ideas.md
│   └── paper_notes/
└── backend/
    ├── pyproject.toml              # Python 包配置
    ├── tests/                      # 测试用例
    └── research_engine/            # 主包
        ├── __main__.py             # CLI 入口
        ├── main.py                 # FastAPI 应用
        ├── config.py               # 配置系统
        ├── models.py               # 数据模型
        ├── schemas.py              # API Schema
        ├── prompts/                # LLM Prompt 模板
        ├── stages/                 # 五阶段实现
        ├── paper_collectors/       # 论文采集器
        └── services/               # 服务层
            ├── llm.py              # LLM 调用服务
            ├── pipeline.py         # 流水线编排
            ├── output.py           # 输出归档
            ├── events.py           # 事件总线
            └── knowledge_base.py   # 知识库加载
```

## 技术栈

- Python 3.11+
- FastAPI + uvicorn（可选 API 模式）
- httpx（HTTP 请求）
- PyYAML（配置加载）
- Pydantic（数据验证）

## LLM 配置说明

系统支持任何 OpenAI Chat Completions 兼容的 API。`llm.provider` 字段仅作为标识名，不影响实际调用逻辑。只要提供了 `base_url`、`api_key` 和 `model`，系统就会使用 OpenAI 兼容协议调用。

常见 LLM 配置示例：

```yaml
# 智谱 GLM
llm:
  provider: "zhipu"
  base_url: "https://open.bigmodel.cn/api/coding/paas/v4"
  api_key: "your-zhipu-api-key"
  model: "glm-4"

# DeepSeek
llm:
  provider: "deepseek"
  base_url: "https://api.deepseek.com/v1"
  api_key: "your-deepseek-api-key"
  model: "deepseek-chat"

# OpenAI
llm:
  provider: "openai"
  base_url: "https://api.openai.com/v1"
  api_key: "your-openai-api-key"
  model: "gpt-4o"
```

如果未配置 LLM（或 API 不可用），系统会自动降级为 heuristic 模式运行，Stage 1/4 仍能基于配置信息产出基础结果。

## 配置参考

详见 [research_config.yaml.example](research_config.yaml.example)。

`pipeline.resume_from` 表示“从指定 stage 开始重新执行”，会复用更早阶段的已完成输出。
`pipeline.continue_from` 表示“沿某次历史调研继续”，可填 session 名或路径，系统会自动定位第一个未完成 stage 并从该处继续。
`paper_search.request_max_retries` 表示论文外部接口的最大重试次数，默认 `3`，重试等待时间按 `10s / 20s / 40s ...` 倍增。
