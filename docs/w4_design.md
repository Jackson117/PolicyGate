# W4 消融设计 — 已定决策

这里只记**已经拍板**的设计选择和理由。数字一律等 W4 跑出来再写。

## 臂的构成：one enabled at a time

```
baseline                       所有机制关
full                           所有已实现机制开
only_<switch>  × 7             baseline + 单个机制
```

7 个开关：`gating`、`context_pruning`、`schema_check`、`value_canonicalization`、
`referent_uniqueness`、`precondition_check`、`parallel_call_block`。
由 `HarnessConfig.ablation_switches()` 自动推出，用 `HarnessConfig.only()` 构造，不手写。

**为什么不用留一法**：几个机制互相兜底——gating 的 scope 规则和 validation 的
referent 检查都能挡住「写错订单」。留一法里去掉任何一个，另一个补上，两行都接近 0，
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
