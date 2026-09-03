# W1 — 可行性验证记录

状态：**任务 1–5 完成，任务 6（baseline 跑批）待执行，卡在 API key。**
本文件里目前**没有任何实验数字**。跑出来之前不写。

---

## 1. 环境（已验证）

| 项 | 值 |
|---|---|
| Python | 3.12.3（`/usr/local/bin/python3`，venv 在 `.venv/`） |
| uv | 0.12.9（homebrew 装的） |
| tau2 pin | `a2c024725189473d2d7cea3a5cfdbcc67478e41f`，2026-08-18，版本串 `tau2 1.0.1` |
| 安装 | `uv sync --extra bench --extra dev`（`bench` = `tau2[knowledge,experiments]`） |
| `tau2 check-data` | ✅ 通过 |

任务量核对（跑 `get_tasks` 实测，非引用文档）：
retail **114**、banking_knowledge **97**、airline **50**、telecom **2285**。
前两个与 CLAUDE.md 一致；airline 和 telecom 的数字 CLAUDE.md 没写，记在这里备用。

### 踩到的坑：iCloud Drive 同步 `~/Documents`，editable 安装静默失效

**已解决——项目已从 `~/Documents/Code/PolicyGate` 移到 `~/Code/PolicyGate`。**
下面保留记录，因为排查过程绕了弯路，而且这个坑对任何放在 `~/Documents`
下的 Python 项目都成立。

现象：`uv sync` 刚结束时 `import tau2` 正常，几分钟后同一条命令开始报
`ModuleNotFoundError`，而 venv 里一切看起来都正常，`.pth` 文件内容也正确。

机制：iCloud 给它接管的文件打 macOS `UF_HIDDEN` 标志，而 CPython 的
`site.addpackage()` **静默跳过隐藏的 `.pth` 文件**——不报错、不警告，
editable 安装就这么消失了。

**第一次归因错了**，当时写的是「端点/备份类 agent 在盖章」，并靠
`chflags nohidden` 绕过去。那个结论站不住，两条证据推翻它：

1. 新建的文件**不带** hidden 标志——不是有人在无差别盖章
2. 出现了 iCloud 特征的冲突副本：`_editable_impl_policygate 2.pth`、
   `uv_cache 2.json`、`direct_url 2.json`、`uv_build 2.json`，
   以及带 `dataless` 标志（已驱逐到云端）的文件

`~/Library/Mobile Documents/com~apple~CloudDocs/Documents` 存在，
说明「桌面与文稿」同步是开着的。

而且 `chflags` 方案后来**彻底失效**了——清完标志同一秒再查就又是 hidden，
iCloud 抢得比脚本快。

`.pth` 只是最先炸的地方。放在 iCloud 里还会有：`.venv` 上千个文件持续同步、
`third_party/tau2-bench` 209MB 白占空间、`.git` 被插一脚可能产生冲突副本、
文件被驱逐成 `dataless` 导致跑批中途卡住。

搬到 `~/Code/` 之后全部消失：`.pth` 标志干净、无冲突副本、
裸调用 `tau2` 不需要 `PYTHONPATH`。绕行用的
`scripts/fix_editable_installs.sh` 和 Makefile 里的 `PYTHONPATH` 都已删除。

⚠️ 你 `~/Documents/Code/` 下还有约 35 个其它项目，同样在被 iCloud 同步。

---

## 2. tau2 契约（读源码确认，非推测）

- agent 基类 `HalfDuplexAgent`，实现 `get_init_state` / `generate_next_message`。
  与 CLAUDE.md 记的一致。
- **消息格式硬约束**（`base_agent.py: validate_message_format_default`）：
  一条 `AssistantMessage` 只能**要么**有文本内容**要么**有 tool call，
  两者都有或都没有都判非法。
  → 直接影响审批闸门设计：闸门拦下一个提议的写操作、想跟用户说明时，
  只能返回**纯文本、零 tool call** 的消息。不能一边解释一边调工具。
- 编程入口（不必走 CLI）：
  `tau2.runner` 导出 `run_simulation` / `build_agent` / `build_environment` /
  `build_user` / `build_orchestrator` / `get_tasks` / `run_domain` / `run_tasks`。
  `run_domain(TextRunConfig)` 自带并发、断点续跑、重试——**W4 的跑批不用自己写**。
- 参考实现：`examples/agents/custom_agent_eval.py`（手动装配全流程，最接近我们要的）、
  `react_agent.py`（两段式 prompt）、`src/tau2/agent/llm_agent.py`。
- tau2 已有 `metrics/agent_metrics.py: pass_hat_k` 和 `compute_metrics`。
  **点估计不用重写**；但它不出置信区间，bootstrap CI 仍然是我们的 `eval/passk.py`。
- 每条 `SimulationRun` 自带 `duration` / `agent_cost` / `user_cost` /
  `termination_reason` / `trial` / `seed`。成本和延迟统计有现成字段。

### 种子机制（关系到能不能测波动）

`runner/batch.py:748`：
```python
random.seed(config.seed)
seeds = [random.randint(0, 1000000) for _ in range(config.num_trials)]
```
即：k 轮试验用 k 个**互不相同但由 `config.seed` 完全决定**的种子。
所以「同一批任务跑三轮」测到的确实是纯执行噪声，且整批可复现。这是好消息。

### ⚠️ 默认开着一个隐藏的重采样器

`hallucination_retries` 默认 **3**：检测到用户模拟器幻觉时，
tau2 会**换个种子把这条 simulation 重跑一遍**。

这对 W1 是污染——它正好压掉我们要测的轮间波动，而且额外烧钱。
`scripts/w1_baseline.py` 把它默认设成 **0**，
只在需要复现 tau2 官方发表数字时才调回 3。
`w1_analyze.py` 会检查 `hallucination_retries_used`，非零就报警。

---

## 3. 止损判据需要修正

CLAUDE.md 写的是「pass^1 三轮之间波动在 ±2% 内 → 信号干净」。
**这个判据在 30 任务上测不出来**，理由：

聚合通过率的最小变动步长是 1/30 = **3.3pp**。所以「波动 < 2%」等价于
「轮与轮之间净翻转数为 0」——要么一个任务都没翻，要么翻转恰好互相抵消。
后者会让一个实际很吵的 setup 侥幸通过判据。这个判据既过严又能被巧合骗过。

**正确的仪器是逐任务翻转率**：三轮里结果不一致的任务占比。
它不受聚合抵消影响，直接就是执行噪声的大小。`w1_analyze.py` 报这个数，
并同时报聚合 spread 和分辨率下限（1 任务 = 3.3pp）供对照。

配套的三分类比 pass^1 本身更有信息量：
- **恒失败任务数** = harness 能改善的全部上限（headroom）
- **恒通过任务数** = 只可能被 harness 弄坏，不可能变好
- **不稳定任务数** = 两个 arm 的共同噪声源

如果 headroom 是 2/30，那不管 harness 多好都做不出统计显著的效应——
这才是真正该触发止损的情形，而不是聚合波动超过 2%。

### 另外：比较应该用配对检验

harness arm 和 baseline arm 跑的是**同一批任务**。
用配对检验（McNemar，只看不一致对）而不是比较两个独立的聚合率，
任务难度会被抵消掉，n=30 的功效比表面看起来好得多。
报告里必须给出不一致对的计数，不能只给两个率。

---

## 4. 待办：任务 6

跑批和分析脚本已写好并端到端验证过（用合成数据，非真实结果）：

```bash
make baseline MODEL=<provider/model>
make analyze RESULTS=experiments/<run>/results.json
```

阻塞项：没有 LLM API key，且模型选择直接决定成本。见下节。

---

## 5. 模型与 API 路由（已验证）

走 **AIHubMix** 的 OpenAI 兼容端点。`~/.zshrc` 里只有 `AIHUBMIX_API_KEY`，
**没有 BASE_URL**，而且这些变量不会进到非交互 shell 的环境里，
所以项目根目录写了一份 `.env`（已 gitignore，权限 600）：

```
OPENAI_API_KEY=<AIHUBMIX_API_KEY 的值>
OPENAI_BASE_URL=https://aihubmix.com/v1
OPENAI_API_BASE=https://aihubmix.com/v1
```

tau2 用 python-dotenv 自动读它。模型串一律写成 `openai/<model-id>`。

### 选型：`openai/gpt-4.1-mini`

用 tau2 自己的 `generate()` 路径做过 tool-calling 冒烟测试，
`claude-haiku-4-5` 和 `gpt-4.1-mini` 经这个代理都能返回结构正确的 tool call。
定选后者，理由是一个硬差异：

**`openai/claude-haiku-4-5` 在 litellm 里没有价格映射，`cost` 恒等于 0。**
（litellm 按 `custom_llm_provider=openai` + 型号名查表，查不到就报
`This model isn't mapped yet` 然后返回 0。）
那样 W1 的「单轮成本」这项直接测不出来，止损判据里的成本条款也就失效了。
`gpt-4.1-mini` 有映射，能出数字，且它本来就是 tau2 examples 的默认模型，
function calling 是已知可用的路径。

⚠️ **成本数字的口径**：litellm 按 **OpenAI 官方牌价**折算，
不是 AIHubMix 实际扣费。报告里要写成「按 OpenAI 牌价折算的等价成本」，
并跟 AIHubMix 后台账单对一次，给出换算系数。绝不能当成真实账单直接写进 README。

---

## 6. ⚠️ 修正一条 CLAUDE.md 里的事实：retail 的评分**含 LLM 判分**

CLAUDE.md 写：「评分是世界状态哈希比对，**不是文本比对，不是 LLM 判分**」。
这条对 retail **不成立**。实测每个任务的 `evaluation_criteria.reward_basis`：

| 域 | reward_basis 分布 |
|---|---|
| retail (114) | `DB,NL_ASSERTION` **112**；`DB` 2 |
| banking_knowledge (97) | `DB` **87**；`ACTION` **9**；`DB,NL_ASSERTION` 1 |

即 retail 里 **112/114** 的最终 reward 是 DB 检查 **与** NL 断言判分的**合取**
（`evaluator.py` 里把各 basis 的 reward 相乘/取合），
而 NL 断言由 **LLM 判分**（`evaluator_nl_assertions.py: evaluate_nl_assertions`）。

缓解因素：`nl_assertions` 为空时短路返回 1.0，不发 LLM 请求。
retail 里真正带非空断言的只有 **40/114**；
**W1 用的前 30 个任务里有 9 个**（id `2 3 4 16 19 21 24 28 29`）。

### 为什么这条很要紧

1. **评分链路里有一个非确定性的 LLM**。它给每次测量额外注入噪声，
   而这噪声跟 harness 好不好完全无关。轮间波动里有一部分来自判分器，不是来自 agent。
2. **判分器会挂，而且挂了整条 simulation 作废**。pilot 里 3 条挂了 1 条：
   AIHubMix 返回 `no_available_channel`（400），
   栈顶在 `evaluate_simulation → NLEvaluator.calculate_reward`——
   **对话本身已经跑完并付过费**，tau2 却把整条标成 `INFRASTRUCTURE_ERROR`、
   丢弃全部消息（`duration=0`、`msgs=0`）。钱花了，数据没了。
3. **消融要归因到 DB 分量**。我们三个贡献（状态门控 / 幂等 / 审计）
   影响的是世界状态终态，不是措辞。
   报告必须拆开 **DB 分量** 和 **NL 分量** 分别报，
   否则 harness 的效应会被判分器噪声稀释，看上去比实际小。
   `RewardInfo.reward_breakdown` 里有按 `RewardType` 拆好的分量，直接取。

### 附带确认：banking_knowledge 的 ACTION 任务是这 9 个

CLAUDE.md 提醒过「banking_knowledge 有小部分任务用 `RewardType.ACTION`，
需要筛出单独处理」。实测就是 9 个：
`task_004 task_008 task_012 task_014 task_032 task_033 task_034 task_035 task_083`。

顺带一提，**banking_knowledge 87/97 是纯 DB 评分**，几乎没有 LLM 判分，
比 retail 更贴合本项目的论点。W4 消融的主域选它比选 retail 干净。

---

## 7. 另一个坑：`save_to` 是**运行名**不是路径

`runner/batch.py:1085`：`save_dir = DATA_DIR / "simulations" / (config.save_to or ...)`。
传路径进去会在 pin 死的 tau2 checkout 里造出一个名叫 `results.json` 的**目录**。
`w1_baseline.py` 现在传运行名，拿到 `run_domain` 的返回值后
自己 `results.save()` 到 `experiments/<run>/` —— 结果与配置同目录。

## 8. 单次成本（pilot 实测，n=2 条成功的 simulation）

`openai/gpt-4.1-mini` 双边，retail：
**agent ≈ $0.0104，user ≈ $0.0019，合计 ≈ $0.0123/次，耗时 ≈ 38s。**
（按 OpenAI 牌价折算，非 AIHubMix 实际扣费。）
外推 90 次 ≈ **$1.1**，远低于预算的 1/20。成本这条止损线不触发。

---

## 9. 判分器模型在 AIHubMix 上路由不到（确定性故障，已修）

第一次跑全量时 30 条里 7 条 `INFRASTRUCTURE_ERROR`，且集中在带 nl_assertions 的任务上。
不是随机抖动，是确定性的：

`tau2/config.py:24` 写死
```python
DEFAULT_LLM_NL_ASSERTIONS = "gpt-4.1-2025-04-14"
```
**裸模型名、没有 provider 前缀**。litellm 把它当 OpenAI 模型，
于是走我们配的 `OPENAI_BASE_URL`（AIHubMix）。
而 AIHubMix 目录里只有 `gpt-4.1` / `gpt-4.1-mini` / `gpt-4.1-nano`，
**没有 `gpt-4.1-2025-04-14`**，返回 400 `no_available_channel`。

单独验证过：
```
openai/gpt-4.1        OK
gpt-4.1-2025-04-14    BadRequestError: no_available_channel
```

后果特别恶心：故障发生在 `evaluate_simulation` 阶段，
**对话已经跑完并付过费**，然后整条 simulation 被丢弃。花钱买了个空。

修法：`w1_baseline.py` 加了 `--judge-llm`（默认 `openai/gpt-4.1`）。
注意判分模型是**模块级常量**，且 `evaluator_nl_assertions.py` 在 import 时
按值绑定，所以只改 `tau2.config` 无效，必须重新绑定使用它的模块里的名字：
```python
from tau2.evaluator import evaluator_nl_assertions
evaluator_nl_assertions.DEFAULT_LLM_NL_ASSERTIONS = args.judge_llm
```

⚠️ **可比性声明**：我们的判分模型是 `gpt-4.1`（AIHubMix 路由），
tau2 官方是 `gpt-4.1-2025-04-14`（OpenAI 直连）。
不是同一个快照，也不是同一条链路。
**我们的绝对分数不能跟公开 leaderboard 直接比。**
项目论点是「同模型同用户模拟器下 arm 之间的差」，不依赖绝对值可比性，
但 README 里必须把这句写出来。

同类隐患还有 `DEFAULT_LLM_ENV_INTERFACE = "gpt-4.1-2025-04-14"`（同样裸名）。
retail 没走到它，换域时要留意。

---

## 10. W1 结果与止损判断

`experiments/2026-09-02_w1_baseline/`，retail 前 30 个任务 × 3 轮 = 90 次 simulation。
`openai/gpt-4.1-mini` 双边（temperature=0），判分器 `openai/gpt-4.1`，
`hallucination_retries=0`，seed=42，耗时 17.9 分钟。
**90/90 正常终止，零 infra error。**

### 数字

| | pass^1 | pass^3 | 三轮不一致的任务 | 恒失败 |
|---|---|---|---|---|
| **TOTAL** | **0.522** [0.400, 0.644] | **0.233** [0.100, 0.400] | 17/30 (57%) | 6 |
| DB 分量 | 0.656 | 0.367 | 16/30 (53%) | 3 |
| NL 分量 | 0.822 | 0.767 | 4/30 (13%) | 3 |

逐轮聚合通过率：0.467 / 0.500 / 0.600，spread **13.3pp**。
成本 **$1.14 / 90 次 = $0.0127 每次**，中位耗时 44.4s。

### 判据怎么读

按 CLAUDE.md 字面：spread 13.3pp > ±10%，落在
「噪声压过信号，需重新设计」那一档。**但这个诊断在这里是错的。**

分解结果说明了原因：**噪声几乎全在 DB 分量**（53% 翻转），
LLM 判分器只贡献 13%。也就是说，波动不是测量误差污染了信号——
**波动本身就是被研究的现象**：同一个任务、同一个模型、
temperature=0，agent 每次对世界状态做出不同的写操作。

这正是本项目要修的东西。把它当噪声排除掉，等于把研究对象扔了。

### 真正的 headroom 在 pass^3，不在 pass^1

DB 分量 pass^1 = 0.656 但 pass^3 = 0.367，中间 **29pp 的落差全是不可靠性**。
恒失败的只有 3/30——也就是说 agent「知道怎么做」的任务远多于它「每次都做对」的任务。

一个把行为稳定下来的 harness，理论上限是把 DB pass^3 从 0.367 推到 0.656。
**这个 headroom 比 pass^1 的头顶空间大得多，而且它就是可靠性的定义。**

→ **README 的主指标必须是 pass^3（或 pass^k 曲线），不是 pass^1。**
只报 pass^1 会把项目最强的证据藏起来。

### 结论：继续，但改四处实验设计

**不止损。** 成本线离触发差得远（单次 $0.0127，全量 retail 114×3 一个 cell 才 $4.3），
且波动是信号不是噪声。降级到 AgentDojo 没有理由。

要改的：

1. **任务量从 30 提到 114（retail 全量）**。57% 的翻转率下 n=30 的功效不够；
   钱不是约束（一个 cell $4.3），墙钟才是（并发 4 时约 21 分钟/cell）。
2. **轮数从 3 提到 5**。3 轮时逐任务 pass^1 只能取 {0, ⅓, ⅔, 1} 四个值，
   太粗；5 轮才能给出像样的逐任务可靠性估计，也才能画 pass^k 曲线。
3. **主指标换成 pass^3 / pass^k 曲线，并且 DB 分量与 NL 分量分开报。**
   合起来报会被判分器稀释。
4. **配对 McNemar 检验**，报不一致对的计数。两个 arm 跑同一批任务，
   任务难度可以抵消掉，别浪费这个设计优势。

另外建议把 **banking_knowledge 作为消融主域**：87/97 是纯 DB 评分，
几乎没有 LLM 判分器介入，测量链路比 retail 干净，且业务场景本来就更贴合。
retail 保留作为第二个域，用来说明结论不是单域特例。
