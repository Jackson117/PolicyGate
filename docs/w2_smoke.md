# W2 冒烟运行记录 — 2026-09-13

**目的不是出实验结论，是回答单元测试答不了的问题**：适配器在真实模型下是否真的满足 tau2 契约、
harness 会不会拒绝真实模型提出的东西、审批闸门代价几何。

样本量 n=12、单轮，**任何通过率差异都不具统计意义**，下面会明确标出哪些数字能读、哪些不能。

---

## 1. 跑了什么

| | |
|---|---|
| 脚本 | `scripts/w2_smoke.py` |
| 域 / 模型 | retail，agent 与 user 均 `openai/gpt-4.1-mini`（AIHubMix），temperature 0 |
| 判分器 | `openai/gpt-4.1`（绕开 tau2 写死的不可路由 id，同 W1） |
| 任务选择 | `--only-write-tasks`：参考轨迹里含写操作的任务，取前 12 个 |
| 臂 | `full` 与 `baseline`，**同一套适配器代码**，跑**同一批 12 个任务** |
| 试验次数 | 1 |
| `hallucination_retries` | 0 |

任务选择说明：retail 前 3 个任务是纯读的，审批闸门在不写的任务上永远不会触发，
所以按「参考轨迹含写操作」筛。这是**对任务的筛选，不是对结果的筛选**，
但它仍然让这里的通过率**不能和 W1 跑的前 30 个任务比**。

---

## 2. 契约验证：通过

| 检查 | 结果 |
|---|---|
| 12/12 simulation 正常结束 | ✅ 全部 `USER_STOP`，零 infra error |
| 审计链完整性 | ✅ 12/12 `verify().intact` |
| 消息格式硬约束（文本 xor tool call） | ✅ 零违例——tau2 会直接拒收，没有拒收发生 |
| 预算耗尽强制升级 | 0 次。模型总能在 3 次内部尝试内给出可接受的提议 |
| tau2 内部重试 | `agents_built == simulations == 12`，无隐藏重跑 |

---

## 3. harness 真的在拒绝东西

12 个 session 里共 **49 次拒绝**：

| 规则 | 次数 | 出现在几个 session |
|---|---|---|
| `gating.one_call_per_turn` | 41 | **12 / 12** |
| `validation.refused` | 8 | 4 / 12 |
| `gating.budget_exhausted` | 0 | 0 |

`one_call_per_turn` 在**每一个** session 都触发了。这与 W1 观察一致
（90 条 simulation 里 70 条违反 retail policy.md 第 20 行），
区别是 W1 里这个违规**不被扣分**，这里被 harness 挡住了。

审批闸门：

```
写提议            12
  被上游拦下       8   ← gating / validation 在闸门之前就拒了
  到达闸门         4
  闸门触发         4   → 触发率 1.00
  批准             4   → 通过率 1.00（auto_approve）
实际执行           4
```

**触发率 1.00 是构造出来的，不是测量出来的**：当前触发规则是「所有写操作都暂停」，
所以这个数字恒等于 1，不携带任何信息。它是选择性触发规则的基线，不是结果。

---

## 4. 通过率：配对比较，n=12

| 臂 | 通过 |
|---|---|
| baseline | 4 / 12 |
| full | 3 / 12 |

配对表（同一批任务）：

| | full 过 | full 挂 |
|---|---|---|
| **baseline 过** | 3 | **1** |
| **baseline 挂** | 0 | 8 |

**不一致对只有 1 个。McNemar 在 1 个不一致对上 p = 1.0——这个差异等于没测到。**
不要把「0.25 vs 0.33」写成结论，它是 1 个任务的差别，而 1 个任务在 n=12 上就是 8.3pp。

### 但那 1 个任务的机制值得看

task 9，用户要换两件商品。

- **baseline** 在**一个 turn 里并发发了 3 次 `get_order_details`**，省下 2 个 turn，
  在用户失去耐心之前跑完了写操作，通过。
- **full** 被 `one_call_per_turn` 拦住，三次查询拆成三个 turn，
  等走到确认那一步用户已经 `###STOP###` 了，写操作从未发生。
  assistant turn 数：baseline 14，full 11。

**拒绝本身不消耗 turn**（内部重试对 orchestrator 不可见），
消耗 turn 的是「一次只能调一个工具」这条规则本身。

这条不是我的过度执行。retail `policy.md:20` 原文是
「You should at most make one tool call at a time」，**没有限定只管写操作**，
上面那 3 次并发的是读操作，同样违规。

所以这是一个真实的、值得写进报告的结构性结论：

> **tau2 不为遵守它自己的策略给分，但会因为回合变长而终止对话。
> 于是合规模块在这个基准上被记为纯损失。**

推论：`parallel_call_block_enabled` 这一行在 W4 消融里**很可能是负的**，
而正确的解读不是「这个机制没用」，而是「这个基准的 reward 与它自带的策略不一致」。
报告里必须把**策略违规率**作为与 reward 并列的独立指标报出来，否则这个机制的全部价值都看不见。

---

## 5. 成本：tau2 的账单少算了 harness

| 项 | full | baseline |
|---|---|---|
| tau2 记的 agent 成本 | $0.1166 | $0.1047 |
| agent 实际 LLM 花费（含被拒提议） | $0.1349 | $0.1047 |
| **被隐藏的重试成本** | **$0.0184（+15.7%）** | $0 |
| user simulator | $0.0290 | $0.0283 |

**被拒的提议永远不会进入返回给 tau2 的轨迹，所以 `SimulationRun.agent_cost` 对 harness 臂是个下界，不是账单。**
`scripts/w2_smoke.py` 自己在 `_propose` 上计数，这是上表第二行的来源。

还有一笔**两边都没算**的：NL 断言判分器（`openai/gpt-4.1`）的调用不计入
`agent_cost` 也不计入 `user_cost`。retail 112/114 任务的 reward 含 LLM 判分，
这笔钱目前完全不在账上。

口径同 W1：litellm 按 **OpenAI 牌价**折算，不是 AIHubMix 实际扣费。

本次全部运行（含两次失败的 3 任务试跑）合计约 **$0.33**（牌价折算），远低于 $5 上限。

---

## 6. 过程中踩到的两个坑

### 6.1 AIHubMix 间歇性 `Your account balance is insufficient`

第一次 3 任务试跑 3 条挂了 2 条，第二次挂了 1 条，报错是
`litellm.APIError: OpenAIException - Your account balance is insufficient`。
但紧接着单发一条最小请求**能成功**，12 任务正式跑**零报错**。

所以不是密钥问题，像是 AIHubMix 某条上游 channel 余额耗尽后的间歇路由失败。
**危险之处和 W1 的判分器 bug 同构**：故障发生在对话跑完之后，
钱花了、整条 simulation 被标成 `INFRASTRUCTURE_ERROR` 丢弃。
W4 跑全量前建议先确认账户余额。

### 6.2 `save_to` 撞名会触发交互式 resume 提示

tau2 写到 `DATA_DIR/simulations/<save_to>/`，发现已有文件就**交互式询问**是否续跑，
非交互 shell 下直接 `EOFError` 崩掉——在打印完配置之后、跑任何东西之前。
与 `tau2 review` 覆盖输出文件时的坑是同一个。
`w2_smoke.py` 现在给 `save_to` 加时间戳后缀。

---

## 7. 本次运行的产物与一处不一致

`experiments/2026-09-13_w2_smoke_{full,baseline}/`：`config.json`、`smoke_report.json`
（`results.json` 按 `.gitignore` 不入库）。

⚠️ **`chains.jsonl` 是这次跑完之后才加进脚本的**，所以这两个目录里没有它。
改动是纯增量输出，不影响已产生的数字，但严格说提交的脚本与产出这批结果的脚本不是同一份。
下次跑会带上审计链原始记录。

---

## 8. 结论

- **契约成立**，可以继续往下写。12/12 正常结束、审计链全部完整、零格式违例。
- **harness 确实在拦东西**，且拦的是 W1 里确认存在、但不被扣分的真实违规。
- **通过率差异没测到**（1 个不一致对），不要当结论用。
- 出现了一个必须带进 W4 设计的结构性问题：**合规与 reward 在这个基准上不对齐**，
  所以策略违规率必须独立报告。
- 当前触发规则（所有写都暂停）让「闸门触发率」退化成常数 1。
  要让这个指标有内容，需要一条选择性触发规则——这是尚未决定是否纳入 W2 的事项。
