"""SimplyPutClaude: does experience-assigned routing protect old reaches?

Protocol (fixed before the main runs, see README):
  1. Learning rates are chosen per controller on a single-task run (bell only,
     seed 999, never used again). Forgetting plays no part in the choice.
  2. Each controller learns bell -> switch -> light -> horn, 1000 practice
     reaches each, from random postures. After every phase, all four tasks are
     evaluated on 200 fixed postures, with learning switched off.
  3. 10 seeds per controller.
  4. Capacity sweep: RouteNet with 4..64 routes (5 seeds).
  5. Reference: plain net practising all four tasks mixed (same total reaches).

Writes results/results.json, results/SUMMARY.md and two plots.
"""

from __future__ import annotations

import json
import os
import time

import numpy as np

from simplyput import arm
from simplyput.models import features, make
from simplyput.protocol import evaluate, interleaved_run, practice, sequential_run

OUT = "results"
EPISODES = 1000
N_EVAL = 200
SEEDS = list(range(10))
SWEEP_SEEDS = list(range(5))
LR_GRID = {
    "plain": [0.001, 0.003, 0.01, 0.03, 0.1],
    "gated": [0.001, 0.003, 0.01, 0.03, 0.1],
    "route": [0.1, 0.3, 1.0, 1.5],
    "route_cueblind": [0.1, 0.3, 1.0, 1.5],
}
BRANCH_SWEEP = [4, 8, 16, 32, 64]


def choose_lr(name: str) -> tuple[float, dict]:
    scores = {}
    for lr in LR_GRID[name]:
        rng = np.random.default_rng(999)
        model = make(name, rng, lr=lr)
        practice(model, rng, "bell", EPISODES)
        postures = arm.random_postures(np.random.default_rng(999_999), N_EVAL)
        scores[lr] = evaluate(model, postures, ["bell"])["bell"]["corrections"]
    best = min(scores, key=scores.get)
    return best, scores


def matrix(run) -> np.ndarray:
    """rows: phase (nothing, bell, switch, light, horn); cols: task."""
    return np.array(
        [[ph["eval"][t]["corrections"] for t in arm.TASK_ORDER] for ph in run["phases"]]
    )


def route_overlap(name: str, seed: int, lr: float) -> np.ndarray:
    """Jaccard overlap of the route sets each task uses, after all four phases."""
    rng = np.random.default_rng(seed)
    model = make(name, rng, lr=lr)
    for task in arm.TASK_ORDER:
        practice(model, rng, task, EPISODES)
    postures = arm.random_postures(np.random.default_rng(10_000 + seed), N_EVAL)
    used = []
    for c, _ in enumerate(arm.TASK_ORDER):
        s = set()
        for th in postures:
            s.update(model.routes(features(th, c)).tolist())
        used.append(s)
    n = len(used)
    J = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            J[i, j] = len(used[i] & used[j]) / len(used[i] | used[j])
    return J


def main():
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    results = {"protocol": {"episodes_per_task": EPISODES, "n_eval": N_EVAL, "seeds": SEEDS,
                            "tasks": arm.TASK_ORDER, "contact_radius": arm.CONTACT_RADIUS,
                            "max_corrections": arm.MAX_CORRECTIONS}}

    # 1. learning rates
    lrs = {}
    results["lr_selection"] = {}
    for name in LR_GRID:
        lr, scores = choose_lr(name)
        lrs[name] = lr
        results["lr_selection"][name] = {"chosen": lr, "bell_only_corrections": scores}
        print(f"lr {name}: {lr}  {scores}")

    # 2-3. sequential runs
    seq = {}
    for name in LR_GRID:
        runs = [sequential_run(name, s, EPISODES, N_EVAL, {"lr": lrs[name]}) for s in SEEDS]
        seq[name] = {"runs": runs, "n_params": runs[0]["n_params"]}
        print(f"{name} done ({time.time()-t0:.0f}s)")

    # 5. interleaved reference
    inter = [interleaved_run("plain", s, EPISODES, N_EVAL, {"lr": lrs["plain"]}) for s in SEEDS]

    # summarise
    summary = {}
    for name, d in seq.items():
        M = np.stack([matrix(r) for r in d["runs"]])  # seeds x phases x tasks
        own = np.stack([M[:, i + 1, i] for i in range(4)], axis=1)  # right after learning task i
        final = M[:, -1, :]
        bell_kept = M[:, -1, 0] / M[:, 1, 0]
        summary[name] = {
            "n_params": d["n_params"],
            "lr": lrs[name],
            "matrix_mean": M.mean(0).round(3).tolist(),
            "matrix_sd": M.std(0).round(3).tolist(),
            "just_learned_mean": own.mean(0).round(3).tolist(),
            "final_mean": final.mean(0).round(3).tolist(),
            "final_avg": float(final.mean()),
            "final_avg_se": float(final.mean(1).std(ddof=1) / np.sqrt(len(SEEDS))),
            "forgetting_avg": float((final[:, :3] - own[:, :3]).mean()),
            "bell_ratio_final_over_learned": float(bell_kept.mean()),
            "bell_ratio_sd": float(bell_kept.std(ddof=1)),
        }
    Mi = np.stack([[r["phases"][0]["eval"][t]["corrections"] for t in arm.TASK_ORDER] for r in inter])
    summary["plain_interleaved"] = {
        "final_mean": Mi.mean(0).round(3).tolist(),
        "final_avg": float(Mi.mean()),
        "final_avg_se": float(Mi.mean(1).std(ddof=1) / np.sqrt(len(SEEDS))),
    }
    naive = summary["plain"]["matrix_mean"][0]
    summary["no_learning"] = {"final_mean": naive, "final_avg": float(np.mean(naive))}

    # 4. capacity sweep
    sweep = {}
    for K in BRANCH_SWEEP:
        rows = []
        for s in SWEEP_SEEDS:
            r = sequential_run("route", s, EPISODES, N_EVAL, {"lr": lrs["route"], "branches": K,
                                                              "active": min(4, K)})
            M = matrix(r)
            rows.append([M[1, 0], M[-1, 0], M[-1].mean()])
        rows = np.array(rows)
        sweep[K] = {"bell_learned": float(rows[:, 0].mean()), "bell_final": float(rows[:, 1].mean()),
                    "final_avg": float(rows[:, 2].mean())}
        print(f"branches {K}: {sweep[K]}")

    overlap = np.mean([route_overlap("route", s, lrs["route"]) for s in SWEEP_SEEDS], axis=0)
    overlap_blind = np.mean([route_overlap("route_cueblind", s, lrs["route_cueblind"])
                             for s in SWEEP_SEEDS], axis=0)

    results["summary"] = summary
    results["capacity_sweep"] = sweep
    results["route_overlap_jaccard"] = overlap.round(3).tolist()
    results["route_overlap_jaccard_cueblind"] = overlap_blind.round(3).tolist()
    results["raw_matrices"] = {n: [matrix(r).round(3).tolist() for r in d["runs"]] for n, d in seq.items()}
    results["runtime_s"] = round(time.time() - t0, 1)

    with open(os.path.join(OUT, "results.json"), "w") as f:
        json.dump(results, f, indent=1)
    write_summary(results)
    plot(results)
    print(open(os.path.join(OUT, "SUMMARY.md")).read())


def write_summary(res):
    S = res["summary"]
    tasks = arm.TASK_ORDER
    L = ["# Results", "",
         f"Corrections needed to touch the target (lower is better; {res['protocol']['max_corrections']} = gave up). "
         f"Mean over {len(res['protocol']['seeds'])} seeds, 200 test postures each.", "",
         "## After learning all four, in order bell → switch → light → horn", "",
         "| Controller | params | lr | " + " | ".join(tasks) + " | average | bell kept (final/just learned) |",
         "|---|---:|---:|" + "---:|" * len(tasks) + "---:|---:|"]
    for name in ["plain", "gated", "route", "route_cueblind"]:
        s = S[name]
        L.append(f"| {name} | {s['n_params']} | {s['lr']} | " +
                 " | ".join(f"{v:.2f}" for v in s["final_mean"]) +
                 f" | **{s['final_avg']:.2f}** ± {s['final_avg_se']:.2f} | "
                 f"{s['bell_ratio_final_over_learned']:.2f} ± {s['bell_ratio_sd']:.2f} |")
    s = S["plain_interleaved"]
    L.append("| plain, all tasks mixed (reference) | | | " + " | ".join(f"{v:.2f}" for v in s["final_mean"]) +
             f" | {s['final_avg']:.2f} ± {s['final_avg_se']:.2f} | — |")
    s = S["no_learning"]
    L.append("| no learning | | | " + " | ".join(f"{v:.2f}" for v in s["final_mean"]) +
             f" | {s['final_avg']:.2f} | — |")
    L += ["", "## Right after learning each task (before later tasks)", "",
          "| Controller | " + " | ".join(tasks) + " |", "|---|" + "---:|" * len(tasks)]
    for name in ["plain", "gated", "route", "route_cueblind"]:
        L.append(f"| {name} | " + " | ".join(f"{v:.2f}" for v in S[name]["just_learned_mean"]) + " |")
    L += ["", "## Bell over time (corrections on the bell after each phase)", "",
          "| Controller | before | after bell | after switch | after light | after horn |",
          "|---|---:|---:|---:|---:|---:|"]
    for name in ["plain", "gated", "route", "route_cueblind"]:
        col = [row[0] for row in S[name]["matrix_mean"]]
        L.append(f"| {name} | " + " | ".join(f"{v:.2f}" for v in col) + " |")
    L += ["", "## Capacity sweep (RouteNet, 5 seeds)", "",
          "| routes | bell just learned | bell at the end | average at the end |", "|---:|---:|---:|---:|"]
    for K, v in res["capacity_sweep"].items():
        L.append(f"| {K} | {v['bell_learned']:.2f} | {v['bell_final']:.2f} | {v['final_avg']:.2f} |")
    L += ["", "## Which routes each task uses (Jaccard overlap, RouteNet, 5 seeds)", ""]
    for title, key in [("cue reaches the routing", "route_overlap_jaccard"),
                       ("cue-blind routing", "route_overlap_jaccard_cueblind")]:
        L += [f"**{title}**", "", "| | " + " | ".join(tasks) + " |", "|---|" + "---:|" * len(tasks)]
        for t, row in zip(tasks, res[key]):
            L.append(f"| {t} | " + " | ".join(f"{v:.2f}" for v in row) + " |")
        L.append("")
    L += ["## Learning-rate selection (bell only, seed 999)", ""]
    for n, v in res["lr_selection"].items():
        L.append(f"- {n}: chose {v['chosen']} from " +
                 ", ".join(f"{k}→{c:.2f}" for k, c in v["bell_only_corrections"].items()))
    L.append("")
    L.append(f"Runtime {res['runtime_s']} s.")
    with open(os.path.join(OUT, "SUMMARY.md"), "w") as f:
        f.write("\n".join(L) + "\n")


def plot(res):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    S = res["summary"]
    colors = {"plain": "#7a7a7a", "gated": "#c07a2c", "route": "#2a6fb0", "route_cueblind": "#9b4fa0"}
    labels = ["before", "after\nbell", "after\nswitch", "after\nlight", "after\nhorn"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ax = axes[0]
    for name, c in colors.items():
        col = [row[0] for row in S[name]["matrix_mean"]]
        ax.plot(range(5), col, "-o", color=c, label=name)
    ax.set_xticks(range(5), labels)
    ax.set_ylabel("corrections to touch the bell")
    ax.set_title("Is the bell remembered while learning other things?")
    ax.legend(frameon=False)
    ax.grid(alpha=0.25)

    ax = axes[1]
    names = ["plain", "gated", "route", "route_cueblind"]
    vals = [S[n]["final_avg"] for n in names] + [S["plain_interleaved"]["final_avg"]]
    errs = [S[n]["final_avg_se"] for n in names] + [S["plain_interleaved"]["final_avg_se"]]
    cols = [colors[n] for n in names] + ["#3c9a5c"]
    ax.bar(range(5), vals, yerr=errs, color=cols, capsize=4)
    ax.set_xticks(range(5), names + ["plain\n(mixed)"])
    ax.set_ylabel("average corrections, all four tasks, at the end")
    ax.set_title("After learning in sequence")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "forgetting.png"), dpi=130)

    fig, ax = plt.subplots(figsize=(5.5, 4))
    Ks = list(res["capacity_sweep"].keys())
    ax.plot(Ks, [res["capacity_sweep"][k]["bell_learned"] for k in Ks], "-o", label="bell, just learned")
    ax.plot(Ks, [res["capacity_sweep"][k]["bell_final"] for k in Ks], "-o", label="bell, after 3 more tasks")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("number of routes")
    ax.set_ylabel("corrections")
    ax.set_title("Fewer routes → routes must be shared")
    ax.legend(frameon=False)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "capacity.png"), dpi=130)


if __name__ == "__main__":
    main()
