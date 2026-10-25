"""Part 3: the flow loop. Rules in PREREGISTRATION.md (Part 3). Writes results_flow/."""

from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

from simplyput.flow import feature_pools, life, make_world

OUT = "results_flow"
SEEDS = list(range(5))
READERS = ["R0", "R1", "R2", "R3"]
POLICIES = ["random", "fixed", "curious", "greedy"]


def se(x):
    x = np.asarray(x, float)
    return float(x.std(ddof=1) / np.sqrt(len(x)))


def beats(a, b):
    return bool(np.mean(a) - np.mean(b) > 2 * np.hypot(se(a), se(b)))


def main():
    try:
        sys.stdout.reconfigure(errors="replace")
    except Exception:
        pass
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    runs = {r: {p: [] for p in POLICIES} for r in READERS}
    for seed in SEEDS:
        echo, _ = make_world(seed)
        for reader in READERS:
            pools = feature_pools(echo, reader, seed)
            for policy in POLICIES:
                runs[reader][policy].append(life(seed, reader, policy, pools=pools))
        print(f"seed {seed} done ({time.time() - t0:.0f}s)")

    summary = {r: {p: {
        "test_acc": float(np.mean([x["test_acc"] for x in runs[r][p]])),
        "test_acc_se": se([x["test_acc"] for x in runs[r][p]]),
        "twin_acc": float(np.mean([x["twin_acc"] for x in runs[r][p]])),
        "pair_acc": float(np.mean([x["pair_acc"] for x in runs[r][p]])),
        "coverage": float(np.mean([x["coverage"] for x in runs[r][p]])),
    } for p in POLICIES} for r in READERS}

    acc = lambda r, p: [x["test_acc"] for x in runs[r][p]]
    rules = {
        "F1_order_needs_joint_reader": bool(
            summary["R1"]["curious"]["twin_acc"] <= 0.65 and summary["R2"]["curious"]["twin_acc"] <= 0.65
            and summary["R3"]["curious"]["twin_acc"] - summary["R2"]["curious"]["twin_acc"] > 0.25),
        "F2_loop_beats_random": beats(acc("R3", "curious"), acc("R3", "random")),
        "F3_loop_beats_best_fixed_pair": beats(acc("R3", "curious"), acc("R3", "fixed")),
        "F4_curiosity_needed": beats(acc("R3", "curious"), acc("R3", "greedy")),
    }
    curves = {p: np.mean([x["curve"] for x in runs["R3"][p]], axis=0).tolist() for p in POLICIES}
    doc = {"seeds": SEEDS, "summary": summary, "rules": rules, "curves_R3": curves,
           "fixed_pairs": [x["fixed_pair"] for x in runs["R3"]["fixed"]],
           "per_seed_test_acc": {r: {p: acc(r, p) for p in POLICIES} for r in READERS},
           "runtime_s": round(time.time() - t0, 1)}
    with open(os.path.join(OUT, "results.json"), "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    write_summary(doc)
    plot(doc)
    print(open(os.path.join(OUT, "SUMMARY.md"), encoding="utf-8").read())


def write_summary(doc):
    S = doc["summary"]
    L = ["# Part 3 results: the flow loop", "",
         f"{len(doc['seeds'])} seeds; 8 hidden objects in 4 twin pairs; 2 probes per episode; "
         "600 learning episodes, then 400 test episodes with no writing. Chance = 0.125.", "",
         "## Test accuracy (identify the hidden object)", "",
         "| Reader | " + " | ".join(POLICIES) + " |", "|---|" + "---:|" * len(POLICIES)]
    for r in READERS:
        L.append(f"| {r} | " + " | ".join(f"{S[r][p]['test_acc']:.3f} +/- {S[r][p]['test_acc_se']:.3f}"
                                          for p in POLICIES) + " |")
    L += ["", "## Twin accuracy (right object, given the right twin pair was found)", "",
          "| Reader | " + " | ".join(POLICIES) + " |", "|---|" + "---:|" * len(POLICIES)]
    for r in READERS:
        L.append(f"| {r} | " + " | ".join(f"{S[r][p]['twin_acc']:.3f}" for p in POLICIES) + " |")
    L += ["", "## Memory coverage (fraction of object x band cells ever written)", "",
          "| Reader | " + " | ".join(POLICIES) + " |", "|---|" + "---:|" * len(POLICIES)]
    for r in READERS:
        L.append(f"| {r} | " + " | ".join(f"{S[r][p]['coverage']:.2f}" for p in POLICIES) + " |")
    L += ["", "## Rules", ""]
    for k, v in doc["rules"].items():
        L.append(f"- {k}: {'PASS' if v else 'FAIL'}")
    L += ["", "Per-seed test accuracy, R3:", ""]
    for p in POLICIES:
        L.append(f"- {p}: " + ", ".join(f"{a:.3f}" for a in doc["per_seed_test_acc"]["R3"][p]))
    L += ["", f"Oracle fixed pairs per seed: {doc['fixed_pairs']}", "", f"Runtime {doc['runtime_s']} s."]
    with open(os.path.join(OUT, "SUMMARY.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")


def plot(doc):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"random": "#7a7a7a", "fixed": "#3c9a5c", "curious": "#2a6fb0", "greedy": "#c07a2c"}
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
    ax = axes[0]
    k = 40
    for p, c in colors.items():
        y = np.convolve(doc["curves_R3"][p], np.ones(k) / k, mode="valid")
        ax.plot(np.arange(len(y)) + k, y, color=c, label=p)
    ax.axhline(0.125, color="#bbb", lw=1, ls=":")
    ax.set_xlabel("learning episode")
    ax.set_ylabel(f"accuracy (rolling {k})")
    ax.set_title("Learning while probing (reader R3)")
    ax.legend(frameon=False)
    ax.grid(alpha=0.25)

    ax = axes[1]
    S = doc["summary"]
    x = np.arange(len(READERS))
    w = 0.2
    for i, (p, c) in enumerate(colors.items()):
        ax.bar(x + (i - 1.5) * w, [S[r][p]["twin_acc"] for r in READERS], w, color=c, label=p)
    ax.axhline(0.5, color="#bbb", lw=1, ls=":")
    ax.set_xticks(x, ["R0 energy", "R1 spectrum", "R2 2nd-order\nscattering", "R3 joint\ntime-freq"])
    ax.set_ylabel("twin accuracy")
    ax.set_title("Telling twins apart needs order")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "flow.png"), dpi=130)


if __name__ == "__main__":
    main()
