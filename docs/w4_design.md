# W4 消融设计 — 已定决策

这里只记**已经拍板**的设计选择和理由。数字一律等 W4 跑出来再写。

## 范围：只跑 retail

W4 只跑 retail（114 任务）。结论只对 retail 成立。
airline 与 banking_knowledge 都缺整套 adapter（工具角色、事实提取、约束）和做功效分析用的基线；
`RunPlan.domain` 限定为 `"retail"`，代码层面不会误跑其他域。
若日后扩域，优先 banking_knowledge：87/97 任务纯 DB 评分，且正是 CLAUDE.md 描述的退款/争议场景。

## 臂的构成：one enabled at a time

```
baseline                       所有机制关
full                           所有已实现机制开
only_<switch>  × 7             baseline + 单个机制
```

7 个开关：`gating`、`context_pruning`、`schema_check`、`value_canonicalization`、
`referent_check`、`precondition_check`、`parallel_call_block`。
由 `HarnessConfig.ablation_switches()` 自动推出，用 `HarnessConfig.only()` 构造，不手写。

**为什么不用留一法**：几个机制互相兜底——gating 的状态规则和 validation 的
referent 检查都作用在写操作的对象上。留一法里去掉任何一个，另一个补上，两行都接近 0，
读起来就是「两个都没用」。单开设计回答的是「这个机制单独有没有用」，不受遮蔽。

**代价，报告里必须写**：各单开效应**加起来不等于** full − baseline，差额是交互效应。
**主结论是 full vs baseline**；单开臂做归因，与共享 baseline 配对比较，Holm 校正。
单开臂出现零效应或负效应（例如 `parallel_call_block`，冒烟里它在花 turn）照实报告。

`tests/test_adapters/test_tau2_agent.py::TestEachSwitchWorksAlone` 保证每个开关
单独打开时确实改变行为。加开关不加场景，这个测试会失败。

## 不进矩阵的机制

| 机制 | 处理 | 理由 |
|---|---|---|
| `audit_enabled` | 所有臂可开，不进矩阵 | 只观测不改变行为，按构造为 0 |
| `approval_gate_enabled` | **所有臂关闭**，不进矩阵 | 决策器固定 `auto_approve` 时不改变轨迹；决策器种类和触发规则不做变量，以简化实验 |
| `idempotency_enabled` | 关，推到 W3 之后 | W1 零重复写调用，无信号 |
| `classified_retry` / `compensation` / `loop_guards` | 关 | 尚无实现，开着等于贴错标签 |

审批闸门的实现和单元测试保留；它不是 W4 的研究变量。

## 指标

- **reward**：总分，以及 DB 分量单独报（retail 112/114 任务含 LLM 判分，harness 作用在世界状态上）
- **policy.md:20 前半句违规率**（一 turn 多调用）：harness 执行的规则。
  执行它的臂里按构造为 0——证明接线，不证明价值，所以与 reward 并排报
- **policy.md:20 后半句违规率**（文本 + 调用）：**harness 不执行**，作为副作用对照指标。
  冒烟里 full 10/12 vs baseline 6/12，p=0.22，待 W4 样本量判断

## tau2 运行配置

- `enforce_communication_protocol = False`（tau2 默认）。打开会让 W1 那 37/90 条混合消息
  直接以 `AGENT_ERROR` 记 0，与 W1 及 tau2 公开数字不可比
- `hallucination_retries = 0`：隐藏的重采样器，压掉要测的轮间波动
- 判分器 `openai/gpt-4.1`：绕开 tau2 写死的、在 AIHubMix 上不可路由的 id
- 剔除一律报告：tau2 把 harness 自身抛出的异常也记作 `INFRASTRUCTURE_ERROR` 并在自带指标里丢弃，
  所以按臂报告剔除数与异常类型，差异性剔除要标出

## 规模：功效分析（`scripts/w4_power.py`，结果在 `experiments/2026-09-14_w4_power/power.json`）

用 W1 基线数据拟合任务难度分布（beta-binomial，矩估计），模拟配对 sign-flip 检验，
求 80% 功效下能检测到的**最小提升**（MDE，单位是 pass^k 在任务间的期望变化）。
114 任务，9 个臂，成本按冒烟实测 $0.0137/条外推（不含判分器）。

拟合可检验：W1 的恒过 / 恒挂 / 不稳定任务数，观测 7/6/17，拟合期望 7.4/6.2/16.4（reward）；
DB 分量观测 11/3/16，期望 11.4/3.1/15.5。

| 轮数 | 统计量 | 9 臂成本 | 主结论 MDE（α=0.05） | 归因 MDE 下限（α/7） |
|---|---|---|---|---|
| 1 | pass^1 | $14 | +16.5pp | +20.9pp |
| 3 | pass^1 | $42 | +9.3pp | +11.8pp |
| 3 | pass^3 | $42 | +15.0pp | +18.9pp |
| 5 | pass^1 | $70 | +7.3pp | +9.3pp |
| 5 | pass^3 | $70 | +9.7pp | +12.5pp |

（reward 分量；DB 分量数字相近，见 json。）

**必须随数字一起读的限制**：
- W1 跑的是 tau2 自带 `llm_agent`、前 30 个任务；W4 跑本项目适配器、全部 114 个。难度分布是外推
- 效应模型是「每个任务 logit 上同一个平移」。验证类机制的真实效应更可能**集中在少数任务**上
  （例如 task 22 的地址规范化只影响个别任务），这种局部效应的功效会低于表中数字
- 检验内部用带连续性校正的正态近似；与 passk 的精确检验在 α 处决策不一致约 1%，偏保守
- 归因列按 Holm 第一步（α/7）算，是最不利那次比较的功效，**下限**不是典型值

**读法**：一个单开机制若只改变 1–3 个任务，效应约 1–3pp，**任何在预算内的设计都测不出显著性**。
归因臂的价值在于描述性证据——哪些任务翻转、哪条规则触发——而不是 p 值。

---

## 偏差记录

### 偏差 1（2026-09-15）：v1 在 3/6 块后停止，referent 规则改为所有权检查

**发生了什么。** v1（`experiments/w4_ablation/`，plan `250a827dddd2`，commit `399ae55`）
headline 阶段跑完 3 块、60 任务 × 5 轮后停止，花费 $8.50。

**触发停止的是中期结果，改动的依据不是。** 60 任务时 full 的 reward pass^1 为 0.357，baseline 为 0.487，
配对差 −0.13，95% CI [−0.21, −0.06]。这个结果只说明需要查原因，
改代码的依据是下面这些**与结果方向无关的正确性证据**：

- full 的 291 次 `validation.refused` 几乎都是同一条理由：「会话锁定在订单 X，却在写订单 Y」。
  锁定对象来自 `retail_facts`：**最近一次 `get_order_details` 返回的订单**。
  多订单任务中，模型逐个查看订单后，锁定落在最后查的那个上。
- 这条规则拒绝的 127 个不同的（工具, 订单）组合里，**107 个是任务参考轨迹里的动作**。
  以相同参数计的 130 个被拒写调用中，117 个直到对话结束也没有执行。
- 触发过该规则的 113 条 simulation，full 通过率 0.08，同任务上 baseline 为 0.37；
  没触发过的 187 条，分别为 0.524 和 0.557。这是相关性证据
  （多订单任务本来就难），同任务 baseline 只控制了任务难度这一项。
- 规则本身与 policy.md 矛盾：政策允许「同一用户的多个请求」，禁止的是「任何其他用户」的请求。

**改了什么。**
- `referent_uniqueness_enabled` 更名为 `referent_check_enabled`。「订单必须被某次查询返回过」保留，
  「必须是最后查的那个」换成「必须属于已认证用户」（`FailureCode.REFERENT_NOT_OWNED`）。
  retail 全部 165 个针对订单的参考写动作中，作用于他人订单的数为 0。
  用新规则离线重审 v1 中被锁定规则拒绝的 130 个写调用：130 个全部放行。
- 状态未取回时 precondition 仍然拒绝（政策要求先查状态再动作），
  但提示从 `has status=None` 改为「状态尚未取回，先查询」。
- runner 在续跑时比对「决定结果的代码」（harness / adapters / eval 的已提交树、`uv.lock`、
  tau2 checkout 的 commit），代码不同则拒绝续跑；代码未提交则拒绝启动。
  v1 的 plan 指纹不含代码版本，修完再续跑会把两个版本的块悄悄混在一起。

**离线审计过、决定不改的规则**（依据 v1 的 300 条 full simulation）：
- `gating.one_call_per_turn`：拒绝 1041 个不同调用，其中 499 个是参考动作，但被拒的是「同一 turn 内的多个调用」，
  政策明文禁止，模型需要逐个重新提议。它的轮次代价是机制本身的代价，照实报告。
- precondition：命中参考动作的真实个例中，task 37 是订单已被取消（拒绝正确）；
  task 104 的参考轨迹是先改地址、再改 items，符合政策；模型在 full 臂里颠倒了顺序，
  先改 items 再改地址。policy.md 说改 items 之后「不能再修改或取消」，harness 拒绝正确；
  但 tau2 的实现（`"pending" in order.status`）会接受这一步，DB 终态照样能对上。
  即 harness 执行政策、评分器不执行，此类失分报告里单列，不为贴合评分而改。
- `gating` 在写之前要求会话处于 SCOPED 状态：`modify_user_address` 的参考动作执行率
  baseline 17/35，full 15/35，未见明显遮蔽，不改。

**预先承诺（v2 用；已被偏差 2 打破，见下）。** v2 从新目录、全部 114 任务从头跑，不复用 v1 的任何块（包括 baseline）。
v2 开跑后，不论结果方向如何，不再修改 harness、adapters、eval；若再发现正确性问题，
记为偏差 2 并报告其影响，不重跑。v1 的数据保留，作为「审计链定位出误伤机制」的证据。

### 偏差 2（2026-09-15）：v2 在 2 块后停止，重复观测改为合并；打破了偏差 1 的「不重跑」承诺

**发生了什么。** v2（`experiments/w4_v2/`，plan `737f7e15846c`，commit `c2d39d4`）
跑完 baseline 和 full 各 1 块（20 任务 × 5 轮）后停止，账本花费 $3.43，
另有第 3 块中途停止、未入账的花费。

**发现途径是偏差 1 定下的审计程序，不是结果方向。** 每块结束后，
把 full 臂的拒绝逐条与参考轨迹比对。
- 14 次「状态尚未取回」的 precondition 拒绝，按审计链时间顺序核对：
  **12 次**是模型先成功调用 `get_order_details` 取回了状态，之后又调用 `get_user_details`；
  2 次是确实没取回过，拒绝正确。
- 原因：`get_user_details` 只列订单号不带状态，adapter 用它**整体替换**了先前的订单观测
  （`tau2_agent.py` 原文 `obj if existing is None else obj`，两个分支相同），状态被抹掉。
  v1 同样有这个缺陷，被 scope 规则的误报遮住了。
- 影响：被误拒的 11 个参考写动作，模型重新取回订单后**全部最终执行**。
  代价主要是轮次，而非写被挡死；无法据此给出对 reward 的效应量。
- 其余 5 个 precondition 拒绝逐条核对均正确（订单已发起换货、或 items 已修改后再改地址/支付/取消），
  没有第三类误报。

**当时的中期数字，照实记录**（20 任务，不构成证据）：reward pass^1 full 0.410 / baseline 0.390，
差 +0.020 [−0.110, +0.170]；pass^5 full 0.15 / baseline 0.20，差 1 个任务，区间大面积重叠。

**为什么打破承诺重跑。** 承诺的用意是防止「看结果改代码」。这次缺陷由预定的审计发现、
有逐条证据、修复不涉及任何阈值或规则取舍；而停跑时沉没成本最低。
代价是承诺失信一次，这里如实记下，由读者判断。

**改了什么。** `KnownObject.observed_again`：新观测里有的字段覆盖，没有的保留；
`validation.observe` 供 adapter 与 service 共用。单元测试与端到端回归测试在修复前均失败。

**v3 第一次启动作废（没有产生新数据）。** 第一次启动时，前两块各花 $0.29、约一分钟就「完成」，
agent 花费为 0，baseline 结果文件与 v2 的逐字节同样大。原因：tau2 的 `auto_resume` 检查点目录按
「plan 名 + plan 指纹」命名，v2 与 v3 的 plan 相同，指纹相同，于是 v3 直接载入了 v2 的模拟。
runner 的代码身份检查只守住了输出目录，没守住 tau2 这一侧。立即停止；该目录移出仓库、不使用。
修复：每个输出目录在认领时生成 `run_id` 写入 plan.json，tau2 检查点按 `run_id` 命名；
两个目录跑同一 plan 的检查点名不同，同一目录续跑时名称不变。v1→v2 没有踩到，是因为开关改名改变了指纹。

**v3 的承诺（替代上面的 v2 承诺）。** v3 从新目录、全部 114 任务从头跑。
v3 开跑后**不再为任何原因重跑**：审计照常进行，发现的问题记为偏差并量化影响，
结论附带这些已知缺陷一起报告。


