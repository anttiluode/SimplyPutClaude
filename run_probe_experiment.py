"""Part 2: Sol's probe-learning world with local vs shared readouts.

Rules are in PREREGISTRATION.md (Part 2). Writes results_probe/.
"""

from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

from simplyput.probe import TASKS, experiment

OUT = "results_probe"
SEEDS = list(range(5))
MODELS = ["shared", "shared_big", "local", "local_cueblind"]
WORLDS = ["anchored", "relative"]
SOL_FROZEN = {"bell_learned": 0.029, "bell_final": 0.714, "final": 0.395}


def dist(run, phase, task):
    return run["phases"][phase]["tasks"][task]["distance"]


def stats(runs):
    bl = np.array([dist(r, 1, 0) for r in runs])
    bf = np.array([dist(r, 4, 0) for r in runs])
    fin = np.array([np.mean([dist(r, 4, k) for k in range(4)]) for r in runs])
    contact = np.array([np.mean([r["phases"][4]["tasks"][k]["contact"] for k in range(4)]) for r in runs])
    own = np.array([[dist(r, k + 1, k) for k in range(4)] for r in runs]).mean(0)
    bell_curve = [float(np.mean([dist(r, p, 0) for r in runs])) for p in range(5)]
    return {
        "parameters": runs[0]["parameters"],
        "bell_learned": float(bl.mean()),
        "bell_final": float(bf.mean()),
        "final": float(fin.mean()),
        "final_se": float(fin.std(ddof=1) / np.sqrt(len(runs))),
        "final_contact": float(contact.mean()),
        "just_learned_each": own.round(3).tolist(),
        "final_each": [float(np.mean([dist(r, 4, k) for r in runs])) for k in range(4)],
        "bell_curve": bell_curve,
    }


def retained(s):
    return s["bell_final"] <= max(1.5 * s["bell_learned"], s["bell_learned"] + 0.05)


def better(a, b):
    return a["final"] < b["final"] - 2 * np.hypot(a["final_se"], b["final_se"])


def main():
    try:
        sys.stdout.reconfigure(errors="replace")
    except Exception:
        pass
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    runs = {w: {m: [experiment(m, s, w) for s in SEEDS] for m in MODELS} for w in WORLDS}
    summary = {w: {m: stats(runs[w][m]) for m in MODELS} for w in WORLDS}

    p0 = stats(runs["anchored"]["shared"][:3])
    rules = {}
    rules["P0_reproduces_sol"] = {
        "pass": bool(all(abs(p0[k] - SOL_FROZEN[k]) <= 0.005 for k in SOL_FROZEN)),
        "measured": {k: round(p0[k], 4) for k in SOL_FROZEN}, "sol": SOL_FROZEN}
    for w in WORLDS:
        S = summary[w]
        rules[w] = {
            "P1_learning_survives": bool(S["local"]["bell_learned"] <= max(2 * S["shared"]["bell_learned"], 0.06)),
            "P2_retention": bool(retained(S["local"])),
            "P3_beats_shared": bool(better(S["local"], S["shared"])),
            "P4_cueblind_fails_retention": bool(not retained(S["local_cueblind"])),
            "P5_beats_param_matched_shared": bool(better(S["local"], S["shared_big"])),
        }
    doc = {"protocol": {"seeds": SEEDS, "episodes_per_task": 650, "eval_postures_per_task_per_wind": 80,
                        "feedback": "two scalar sensor readings per update (Sol's paired probes)"},
           "summary": summary, "rules": rules, "runs": runs, "runtime_s": round(time.time() - t0, 1)}
    with open(os.path.join(OUT, "results.json"), "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    write_summary(doc)
    plot(doc)
    print(open(os.path.join(OUT, "SUMMARY.md"), encoding="utf-8").read())


def write_summary(doc):
    L = ["# Part 2 results: learning from scalar probes, shared vs local readouts", "",
         "First-reach distance to the target (lower is better; contact at 0.12). "
         f"{len(SEEDS)} seeds, 650 probe episodes per task, 80 test postures x 2 winds.", ""]
    r0 = doc["rules"]["P0_reproduces_sol"]
    L += [f"**Reproduction of SimplyPutSol v1 (frozen cable, seeds 0-2):** measured {r0['measured']}, "
          f"published {r0['sol']} -> {'PASS' if r0['pass'] else 'FAIL'}", ""]
    for w in WORLDS:
        S = doc["summary"][w]
        L += [f"## World: {w}", "",
              "| Readout | params | bell just learned | bell after 3 more | final, 4 tasks | final contact |",
              "|---|---:|---:|---:|---:|---:|"]
        for m in MODELS:
            s = S[m]
            L.append(f"| {m} | {s['parameters']} | {s['bell_learned']:.3f} | {s['bell_final']:.3f} | "
                     f"{s['final']:.3f} +/- {s['final_se']:.3f} | {100*s['final_contact']:.0f}% |")
        L += ["", "Bell distance after each phase:", "",
              "| Readout | untrained | bell | switch | light | horn |", "|---|---:|---:|---:|---:|---:|"]
        for m in MODELS:
            L.append(f"| {m} | " + " | ".join(f"{v:.3f}" for v in S[m]["bell_curve"]) + " |")
        L += ["", "Rules:", ""]
        for k, v in doc["rules"][w].items():
            L.append(f"- {k}: {'PASS' if v else 'FAIL'}")
        L.append("")
    L.append(f"Runtime {doc['runtime_s']} s.")
    with open(os.path.join(OUT, "SUMMARY.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")


def plot(doc):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"shared": "#7a7a7a", "shared_big": "#3c9a5c", "local": "#2a6fb0", "local_cueblind": "#9b4fa0"}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, w in zip(axes, WORLDS):
        for m in MODELS:
            ax.plot(range(5), doc["summary"][w][m]["bell_curve"], "-o", color=colors[m], label=m)
        ax.set_xticks(range(5), ["untrained", "after\nbell", "after\nswitch", "after\nlight", "after\nhorn"])
        ax.set_title(f"Bell distance, {w} world")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("first-reach distance to bell")
    axes[0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "probe_forgetting.png"), dpi=130)


if __name__ == "__main__":
    main()
