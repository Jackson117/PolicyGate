from tau2.data_model.simulation import Results
from tau2.metrics.agent_metrics import prepare_dfs

res = Results.load("../experiments/2026-09-02_w1_baseline/results.json")
dfs, df_pass_hat_k = prepare_dfs(res)

flip = df_pass_hat_k[(df_pass_hat_k['pass^1'] > 0) & (df_pass_hat_k['pass^1'] < 1)]
flip_index = flip.index.tolist()

dfs_flip = dfs[dfs['task_id'].isin(flip_index)][['task_id', 'trial', 'success']]

pairs = []
for t_id,g in dfs_flip.groupby('task_id'):
    f = g[~g['success']]
    t = g[g['success']]
    pairs.append([t_id, int(f.iloc[0]['trial']), int(t.iloc[0]['trial'])])


import json

def get_sim(res, task_id, trial):
    for s in res.simulations:
        if s.task_id == task_id and s.trial == trial:
            return s
    return None

def render(sim, max_content=400):
    out = [f"=== task {sim.task_id} / trial {sim.trial} / reward {sim.reward_info.reward} ===",
           f"终止: {sim.termination_reason} | 耗时 {sim.duration:.1f}s\n"]
    step = 0
    for m in (sim.messages or []):
        role = getattr(m, "role", "?")
        if role == "user":
            out.append(f"[用户] {(m.content or '')[:max_content]}")
        elif role == "assistant":
            step += 1
            if m.content:
                out.append(f"[{step} agent说] {m.content[:max_content]}")
            for tc in (m.tool_calls or []):
                out.append(f"[{step} 调用] {tc.name}({json.dumps(tc.arguments, ensure_ascii=False)})")
        elif role == "tool":
            flag = " ⚠ERROR" if m.error else ""
            out.append(f"      ↳ 返回{flag}: {(m.content or '')[:max_content]}")
    return "\n".join(out)

def show_pairs(res, task_id, ok_trial, fail_trial):
    ok_sim = get_sim(res, task_id, ok_trial)
    fail_sim = get_sim(res, task_id, fail_trial)
    print(render(ok_sim))
    print("\n" + "─"*70 + "\n")
    print(render(fail_sim))

from pathlib import Path
Path("failures.txt").write_text(
    "\n\n".join(
        render(get_sim(res, t, f)) + "\n" + "─"*60 + "\n" + render(get_sim(res, t, s))
        for t, f, s in pairs
    ), encoding="utf-8"
)

def diff_args(a_call, b_call):
    ja, jb = json.loads(a_call[1]), json.loads(b_call[1])
    keys = set(ja) | set(jb)
    return {k: (ja.get(k), jb.get(k)) for k in keys if ja.get(k) != jb.get(k)}

def calls_with_context(sim):
    """返回 [(tool_name, args_json, 此调用前最后一条用户消息)]"""
    out, last_user = [], ""
    for m in (sim.messages or []):
        role = getattr(m, "role", None)
        if role == "user":
            last_user = m.content or ""
        elif role == "assistant":
            for tc in (m.tool_calls or []):
                out.append((tc.name, json.dumps(tc.arguments, sort_keys=True), last_user))
    return out

for tid, ft, st in pairs:
    a = calls_with_context(get_sim(res, tid, ft))
    b = calls_with_context(get_sim(res, tid, st))
    n = min(len(a), len(b))
    i = next((j for j in range(n) if a[j][:2] != b[j][:2]), n)
    if i >= n: continue
    if a[i][0] == b[i][0]:
        print(f"--- task {tid} 步{i} {a[i][0]}")
        print("   ", diff_args(a[i], b[i]))