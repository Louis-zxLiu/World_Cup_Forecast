# World Cup Forecast

> 面向足球赛事的可解释多智能体预测平台 —— 把 Elo/泊松统计模型、七个专项 AI 智能体、实时赔率抓取和 LLM 推理链整合成一个可本地部署的一体化工具。

适合：对足球量化分析感兴趣的开发者、体育数据研究者、以及想直观看到"AI 如何一步步推理一场比赛"的技术爱好者。

## ✨ 项目亮点

- **七个专项智能体并发推理**：实力分析、近期状态、新闻舆情、赔率市场、多空辩论、风险管理，结果通过 SSE 流式推送到前端
- **真实数据驱动**：接入 [martj42/football-data](https://github.com/martj42/international_results) 全量 4.9 万场国际比赛，基于赛事级别 K 因子和进球差倍率计算 336 支球队的真实 Elo
- **统计精度更高**：Dixon-Coles 双泊松模型（rho=-0.13），回测 Brier score 从 0.689 降至 0.584（降幅 15%）
- **LLM 推理链可见**：每个智能体的分析步骤（观察/分析/结论）以结构化 JSON 流式返回，无 LLM key 自动降级为规则推理
- **国内可用的联网搜索**：支持博查 Bocha / 智谱 Web Search，与 LLM 解耦，新闻舆情智能体失败时诚实报告而非假装中性
- **傻瓜式首页**：自然语言输入（"巴西 vs 阿根廷"）、一键今日推荐、Kelly 仓位建议，不需要了解底层参数

## 📸 界面预览

| 首页 / 今日推荐 | 推理追踪 | 系统设置 |
|---|---|---|
| 自然语言提问 + 一键批量预测 | 每个智能体可折叠的 LLM 步骤 | LLM / 搜索配置 + 一键数据初始化 |

## 📦 安装

### 一键启动（Windows）

```bat
start.bat
```

脚本自动创建虚拟环境、安装依赖、启动后端和前端，并打开浏览器。

### 手动安装

```bash
# 1. 后端
python -m venv .venv
.venv/Scripts/python -m pip install -e ".[dev]"
.venv/Scripts/python -m uvicorn apps.api.main:app --reload --port 8000

# 2. 前端（另开终端）
cd apps/web
npm install
npm run dev
```

打开 `http://localhost:5173`。

## 🚀 快速开始

1. **初始化数据**：进入「系统设置」→ 点击「下载并初始化数据」（约 1-2 分钟，仅需一次）
2. **配置 LLM**：填入 OpenAI-compatible API URL、模型名和 key（支持 DeepSeek / 智谱 / OpenAI 等），点「测试连接」后保存
3. **开始预测**：回到首页，输入"巴西 vs 德国"或点「今日推荐」

## ⚙️ 配置说明

### LLM API

支持任何 OpenAI 兼容接口。在「系统设置 → LLM API 配置」填写：

| 字段 | 示例 |
|---|---|
| API Base URL | `https://api.deepseek.com/v1` |
| 模型名 | `deepseek-chat` |
| API Key | `sk-...` |

未配置时，预测和赔率功能正常，智能体解释降级为规则模板。

### 联网搜索（可选）

新闻舆情智能体需要独立配置搜索源（Google News 国内不通）：

- **博查 Bocha**（推荐）：`https://api.bochaai.com/v1/web-search`
- **智谱 Web Search**：`https://open.bigmodel.cn/api/paas/v4/web_search`

在「系统设置 → 联网搜索配置」填入 key 并勾选启用即可。

## 📁 项目结构

```
World_Cup_Forecast/
├── apps/
│   ├── api/main.py          # FastAPI 后端，SSE 流式端点
│   └── web/src/             # React + TypeScript 前端
│       └── components/
│           ├── Home.tsx         # 首页（自然语言 + 今日推荐）
│           ├── Workbench.tsx    # 详细预测工作台
│           ├── ReasoningTrace.tsx  # 智能体推理追踪卡片
│           └── Settings.tsx     # 系统设置
├── packages/worldcup_forecast/
│   ├── agents.py            # 七个专项智能体
│   ├── modeling.py          # Elo + Dixon-Coles 双泊松模型
│   ├── ingest.py            # 数据下载与 Elo 计算
│   ├── reasoning.py         # LLM 结构化推理链
│   ├── search.py            # 可插拔联网搜索层
│   ├── form.py              # 近期状态统计
│   ├── storage.py           # DuckDB 数据层
│   └── schemas.py           # Pydantic 数据模型
├── tests/                   # pytest 测试套件（54 个用例）
└── start.bat                # Windows 一键启动
```

## 🔌 主要 API 端点

| 端点 | 说明 |
|---|---|
| `POST /api/predict/match/stream` | SSE 流式预测，返回智能体推理 + 最终报告 |
| `POST /api/ask` | 自然语言提问，LLM 抽取球队名再预测 |
| `GET /api/predict/today` | 批量预测今日赛程，标注价值投注 |
| `POST /api/ingest/run` | 下载国际比赛数据并重建 Elo |
| `GET/PUT /api/settings/llm` | LLM 配置读写 |
| `GET/PUT /api/settings/search` | 联网搜索配置读写 |
| `GET /docs` | FastAPI 自动生成的交互式文档 |

## 🛠️ 开发

```bash
# 运行测试
.venv/Scripts/python -m pytest

# 代码格式检查
.venv/Scripts/python -m ruff check packages/ apps/ tests/

# 前端类型检查
cd apps/web && npx tsc --noEmit
```

测试覆盖：智能体行为、API 端点、回测指标，共 54 个用例。

## ⚖️ 合规边界

- 爬虫只访问公开页面，不绕过登录、验证码、付费墙或访问控制
- 系统只输出预测概率和风险建议，不自动下注
- 历史数据来源于 [martj42/international_results](https://github.com/martj42/international_results)（开放许可）

## 📝 License

MIT
