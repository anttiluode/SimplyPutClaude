"""Experiment protocol: practice, evaluation, sequential tasks."""

from __future__ import annotations

import numpy as np

from . import arm
from .models import features, make


def task_target(task: str) -> np.ndarray:
    return arm.target_from_posture(arm.TASK_POSTURES[task])


def attempt(model, theta0: np.ndarray, task: str, context: int, learn: bool):
    """One reach + fumble. Returns (corrections, first-reach miss distance)."""
    target = task_target(task)
    x = features(theta0, context)
    dtheta = model.predict(x, context)
    reached_posture = arm.clip_posture(theta0 + dtheta)
    miss = float(np.linalg.norm(target - arm.hand(reached_posture)))
    final, n, ok = arm.fumble(reached_posture, target)
    if learn and ok:
        model.learn(x, context, final - theta0)
    return n, miss


def evaluate(model, postures: np.ndarray, tasks=None):
    tasks = tasks or arm.TASK_ORDER
    out = {}
    for task in tasks:
        c = arm.TASK_ORDER.index(task)
        ns, misses = zip(*(attempt(model, th, task, c, learn=False) for th in postures))
        out[task] = {
            "corrections": float(np.mean(ns)),
            "miss": float(np.mean(misses)),
            "first_try_contact": float(np.mean(np.array(ns) == 0)),
        }
    return out


def practice(model, rng: np.random.Generator, task: str, episodes: int):
    c = arm.TASK_ORDER.index(task)
    for th in arm.random_postures(rng, episodes):
        attempt(model, th, task, c, learn=True)


def sequential_run(model_name: str, seed: int, episodes: int, n_eval: int, hp: dict):
    """Learn bell, then switch, then light, then horn. Evaluate all after each phase."""
    rng = np.random.default_rng(seed)
    eval_rng = np.random.default_rng(10_000 + seed)
    eval_postures = arm.random_postures(eval_rng, n_eval)
    model = make(model_name, rng, **hp)
    phases = [{"after": "nothing", "eval": evaluate(model, eval_postures)}]
    for task in arm.TASK_ORDER:
        practice(model, rng, task, episodes)
        phases.append({"after": task, "eval": evaluate(model, eval_postures)})
    return {"model": model_name, "seed": seed, "n_params": model.n_params(), "phases": phases}


def interleaved_run(model_name: str, seed: int, episodes: int, n_eval: int, hp: dict):
    """Reference: the same total practice, but all four tasks mixed (no sequence)."""
    rng = np.random.default_rng(seed)
    eval_rng = np.random.default_rng(10_000 + seed)
    eval_postures = arm.random_postures(eval_rng, n_eval)
    model = make(model_name, rng, **hp)
    total = episodes * len(arm.TASK_ORDER)
    tasks = rng.integers(0, len(arm.TASK_ORDER), size=total)
    for th, c in zip(arm.random_postures(rng, total), tasks):
        attempt(model, th, arm.TASK_ORDER[c], int(c), learn=True)
    return {
        "model": model_name + "_interleaved",
        "seed": seed,
        "n_params": model.n_params(),
        "phases": [{"after": "all_mixed", "eval": evaluate(model, eval_postures)}],
    }
