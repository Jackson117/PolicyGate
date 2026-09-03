# policygate — 项目文件架构

> 项目名 `policygate` 是占位，取「policy + gating」之意，你可以改。
> 全文假定包名与目录名一致，改名时记得同步 `pyproject.toml`。

---

## 顶层

```
policygate/
├── README.md                 # 门面：英雄图 + 结论 + 方法论章节
├── pyproject.toml            # 依赖与包配置（Python >=3.12,<3.14）
├── docker-compose.yml        # api + postgres + streamlit 一键起
├── Dockerfile
├── Makefile                  # make eval / make ablate / make viz / make test
├── alembic.ini               # 审计库迁移配置
├── .env.example              # API key 占位，绝不提交真 key
├── .github/workflows/ci.yml  # pytest + ruff + 框架依赖检查 + 子集回归
│
├── src/policygate/           # 全部源码
├── configs/                  # 实验配置（yaml）
├── experiments/              # 实验产出（config + 原始结果 + 图）
├── tests/
├── scripts/
├── docs/
└── third_party/tau2-bench/   # pin 死版本的 fork（submodule 或 vendored）
```

`third_party/` 是刻意的：tau2 在活跃开发，内部 API 重构会影响自定义 agent 实现。
fork 一份、pin 死 commit，不跟 main 走。

---

## src/policygate/harness/ — 核心，零框架依赖

**这是项目的技术贡献所在。硬约束：这个目录不许 import LangGraph、LiteLLM、
tau2 或任何 agent 框架，只依赖 Pydantic 和标准库。** CI 里加一条检查强制执行。

这样做的三个理由：它是你的原创部分，交给框架实现等于外包掉唯一的亮点；
纯函数可以脱离 LLM 做单元测试，pytest 秒过；能被多个 adapter 复用。

```
harness/
├── types.py           # 共享数据模型：ToolCall / Decision / CheckResult /
│                      #   ClauseRef / ApprovalRequest / AuditRecord
├── config.py          # HarnessConfig：每个模块一个开关 —— 消融实验的支点
├── state.py           # 会话状态定义与转移规则
│                      #   （unverified → verified → scoped → pending_approval → …）
├── gating.py          # 状态 → 合法工具子集。核心贡献 #1
├── context.py         # 按状态裁剪 policy 文本与 tool schema（同时降 token 成本）
├── guards.py          # 循环守卫：迭代上限 / token 预算 / 墙钟超时 / 重复动作检测
├── validation.py      # Pydantic strict schema 校验 + 前置条件检查，
│                      #   失败返回结构化拒绝而非抛异常
├── idempotency.py     # 幂等键生成、重放检测、首次结果缓存。核心贡献 #2
├── retry.py           # 失败分类：可重试（超时/网络）vs 不可重试（业务规则拒绝）
├── compensation.py    # 写操作失败后的回滚与补偿
├── approval.py        # 审批闸门 + 可插拔审批策略模拟器
│                      #   （auto_approve / auto_reject / oracle / noisy_oracle）
└── audit.py           # 审计记录器 + 哈希链。核心贡献 #3
```

### config.py 值得单独说

消融实验的可行性完全取决于这个文件。每一个 harness 模块必须能独立开关：

```python
class HarnessConfig(BaseModel):
    gating_enabled: bool = True
    context_pruning_enabled: bool = True
    idempotency_enabled: bool = True
    precondition_check_enabled: bool = True
    classified_retry_enabled: bool = True
    compensation_enabled: bool = True
    approval_gate_enabled: bool = True
    loop_guards: LoopGuardConfig
    approval_policy: Literal["auto_approve", "auto_reject", "oracle", "noisy_oracle"]
    seed: int
```

一个开关漏掉，对应的消融行就跑不出来。**动手写模块之前先把这个类定完整。**

### audit.py 的三个属性

审计链不是日志。日志记录发生了什么，审计链**证明**发生了什么且未被篡改：

- **完整性** — 每个改变世界状态的动作都有记录，无缺口
- **关联性** — 每条挂上证据（doc_id + 条款号）、模型推理片段、审批记录、状态 diff
- **可验证顺序** — 追加写、有序、每条带上一条的哈希，篡改即断链

`world_diff` 直接用 `deepdiff`——它已经在 tau2 的核心依赖里，不用额外装。

---

## src/policygate/adapters/ — 接 tau2

```
adapters/
├── tau2_agent.py      # 继承 HalfDuplexAgent，实现：
│                      #   get_init_state(message_history) -> State
│                      #   generate_next_message(message, state) -> (AssistantMessage, State)
├── tool_mapping.py    # tau2 Tool 对象 ↔ harness 内部表示的转换
└── prompts/           # system prompt 模板（按状态组装）
```

**注意 tau2 的执行模型**：orchestrator 拥有外层循环，工具由环境执行不由你执行。
所以审批闸门拦在「模型提议调用」和「你返回 AssistantMessage」之间，
不是拦在「工具执行前」。

先读 `examples/agents/` 和 `src/tau2/agent/llm_agent.py`，你的 adapter 是它的加强版。

---

## src/policygate/service/ — 部署 demo

benchmark 场景不需要 LangGraph（tau2 拥有循环，审批模拟器在进程内）。
**但部署场景需要**：真人审批要跨请求、跨进程恢复，`interrupt()` + Postgres
checkpointer 正是为此设计的。这里也是简历上 LangGraph 关键词的落点。

```
service/
├── graph.py           # LangGraph StateGraph，节点调用 harness/ 里的同一套逻辑
├── api.py             # FastAPI：会话端点 + 审批端点（真人批准/驳回）
├── checkpointer.py    # Postgres checkpointer 配置
└── schemas.py         # 请求/响应模型
```

---

## src/policygate/store/ — 审计持久化

```
store/
├── models.py          # SQLAlchemy：audit_records / approvals / runs / trials
├── repo.py            # 写入与查询
└── migrations/        # alembic
```

**红线**：审计库必须是独立的 Postgres，绝不能碰 tau2 的世界状态 DB。
评分靠世界状态 DB 的哈希比对，你多写一个字节，全部任务判负。

---

## src/policygate/eval/ — 跑批、指标、消融

```
eval/
├── runner.py          # 跑批编排：k 轮试验、并发、断点续跑、结果落盘
├── passk.py           # pass^1 / pass^k 计算 + 置信区间（bootstrap）
├── taxonomy.py        # 失败分类器，八类：
│                      #   选错工具 / 参数错误 / 幻觉 / 工具超时 / 死循环 /
│                      #   越权 / 该升级未升级 / 不该升级却升级
├── ablation.py        # 消融矩阵编排：配置 × 域 × 模型 的笛卡尔积
├── cost.py            # token / 美元 / 延迟统计
└── report.py          # 出图：pass^k 曲线、消融热力图、失败迁移图、成本-可靠性前沿
```

### 两个指标必须分开报告

- **闸门触发率** — harness 决定「要不要问」，这是你的贡献
- **审批通过率** — 模拟器决定「批不批」，这是外生的

只报最终成功率而不报触发率，等于把 oracle 的功劳算在自己头上。
配套还要拆开 **漏放**（该触发未触发，安全事故）和
**过度打扰**（不该触发却触发，体验问题）——绝不能合成一个数字。

---

## src/policygate/viz/ — Streamlit

用 Streamlit 不用 React：纯 Python，两三天能出可用版本，天然适合
「读 trace JSON → 渲染」。别重造 Langfuse 已有的通用 trace 视图，
只做 harness 特有和合规特有的部分。

```
viz/
├── app.py
└── pages/
    ├── 1_paired_trace.py    # 配对对比（主视图）：按分歧点对齐 + 代表性统计
    ├── 2_single_trace.py    # 单条轨迹逐步查看
    ├── 3_ablation.py        # 消融矩阵热力图
    └── 4_failures.py        # 失败分类分布与迁移
```

**配对对比是主视图，单条轨迹是它的降级版。** 单条轨迹回答「这次发生了什么」，
配对回答「我的模块改变了什么」——后者才是消融证据。

每条轨迹旁边必须挂「代表性」元数据（这个模式在 k 轮里出现几次、
同类任务效应方向一致的有几个）。没有它，界面就是个 cherry-picking 机器。

---

## configs/ 与 experiments/

```
configs/
├── baseline.yaml            # 全部 harness 模块关闭
├── full.yaml                # 全部开启
├── ablation/
│   ├── no_gating.yaml
│   ├── no_idempotency.yaml
│   └── ...                  # 每个模块一个
└── models/
    ├── cheap.yaml           # 迭代用：便宜模型 + 小子集
    └── final.yaml           # 出报告用：全量

experiments/
└── 2026-09-15_gating_ablation/
    ├── config.yaml          # 产生这批结果的完整配置（含 seed）
    ├── raw/                 # 原始轨迹 JSON
    ├── metrics.parquet
    └── figures/
```

**结果和产生它的配置必须存在一起。** 这是可复现性的最低要求，
也是你 README 方法论章节的素材来源。

---

## tests/ scripts/ docs/

```
tests/
├── test_harness/       # 纯单元测试，不调 LLM，跑得飞快
├── test_adapters/
└── test_eval/          # pass^k 计算、失败分类器的正确性

scripts/
├── setup_db.sh
├── run_baseline.sh
└── check_no_framework_deps.py   # CI 用：确保 harness/ 不 import 框架

docs/
├── architecture.md     # 本文件
├── methodology.md      # 实验设计：假设、控制变量、样本量、置信区间
└── figures/            # 架构图源文件
```

`docs/methodology.md` 和 CI 里那个依赖检查脚本，是这个项目区别于
普通 agent 作品集的两个具体标志。绝大多数 agent 项目的 README 里没有
「方法」这一节——有这一节本身就是信号。

---

## 构建顺序建议

先写 `types.py` 和 `config.py`（定契约），
再写 `harness/` 各模块 + 单元测试（不需要 LLM，快），
再写 `adapters/tau2_agent.py` 接通，
然后 `eval/` 跑批，
最后 `viz/` 和 `service/`。

viz 放在消融之后是刻意的：先有数据再做可视化，才知道该突出什么。
反过来容易做出一堆没数据填的空壳。
