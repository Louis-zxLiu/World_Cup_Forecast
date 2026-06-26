# World Cup Forecast

可解释、多智能体的世界杯预测与赔率 edge 分析平台。第一版包含 FastAPI 后端、React 前端、500彩票网竞彩足球赔率爬虫、LLM 可配置报告、baseline Elo/Poisson 预测、Kelly 仓位建议。

## Quick Start

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -U pip
.\.venv\Scripts\python -m pip install -e ".[dev]"
.\.venv\Scripts\python -m uvicorn apps.api.main:app --reload --port 8000
```

另开终端：

```powershell
cd apps\web
npm install
npm run dev
```

打开 `http://localhost:5173`。

## 当前能力

- `POST /api/odds/china-lottery/scrape`：抓取并解析 500彩票网竞彩足球赔率，失败时解析本地样例快照。
- `GET/PUT /api/settings/llm`：在前端修改 OpenAI-compatible API URL、模型名、API key、温度、超时。
- `POST /api/predict/match`：输出胜平负概率、期望比分、投注 edge、Kelly 仓位和多智能体解释。
- `POST /api/predict/tournament`：输出简化锦标赛概率。

## 合规边界

爬虫只访问公开页面，不绕过登录、验证码、付费墙或访问控制。系统只输出预测和风险建议，不自动下注。

