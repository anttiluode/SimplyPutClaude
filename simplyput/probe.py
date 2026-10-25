"""Probe learning: Sol's world (SimplyPutSol v1) with local vs shared readouts.

Ported from anttiluode/SimplyPutSol (simplyputsol.py, v1), same constants:
  * Learning sees only two SCALAR sensor readings per update (paired probes
    a +/- sigma*d). No target vector, no correct joint angles.
  * A wind gust at frame t=1 shifts the arm; the final frame t=2 does not show
    it. The right action depends on remembered history.
  * The body is Sol's complex cable reservoir, FROZEN (Sol measured plastic
    and frozen cables as equal, so the readout carries the learning).

The only change under test is the readout:
  shared       one readout B for every task (= Sol's 'frozen' model)
  local        K routes; the final frame picks the routes by resonance
               (tunings start with no cue preference); each route owns its
               readout B_k; a probe updates only the routes it travelled
  local_cueblind  routing ignores the cue (posture only)
  shared_big   one shared readout over a 16x larger frozen reservoir, so its
               readout has as many weights as all 16 local readouts together

Two worlds:
  anchored  Sol's world: action is added to a fixed anchor posture, so the
            start posture does not matter; the answer is one constant per
            (task, wind sign).
  relative  harder: action is added to the arm's CURRENT posture, so the right
            action depends on where the arm starts.
"""

from __future__ import annotations

import numpy as np

D = 9
N_TASKS = 4
TASKS = ("bell", "switch", "light", "horn")
TARGETS = np.array([[-1.10, 1.60], [0.90, 1.00], [0.10, 2.20], [-0.40, 0.60]])
ANCHOR = np.array([-0.1, 1.4])
WIND = np.array([0.35, -0.25])
CONTACT = 0.12
SIGMA = 0.13
LOW, HIGH = np.array([-2.5, 0.2]), np.array([2.5, 2.6])


def hand(q):
    a, b = q
    return np.array([np.cos(a) + 0.8 * np.cos(a + b), np.sin(a) + 0.8 * np.sin(a + b)])


class World:
    def __init__(self, mode: str = "anchored"):
        assert mode in ("anchored", "relative")
        self.mode = mode

    def sense(self, action, task, wind, posture):
        base = ANCHOR if self.mode == "anchored" else posture
        q = np.clip(base + action + wind * WIND, LOW, HIGH)
        distance = float(np.linalg.norm(hand(q) - hand(TARGETS[task])))
        strength = float(np.exp(-0.5 * (distance / 0.85) ** 2))
        return strength, bool(distance <= CONTACT), distance


def frames(posture, task, wind):
    x = np.zeros((3, D))
    cue = np.eye(N_TASKS)[task] * 1.5
    pose = np.array([np.cos(posture[0]), np.sin(posture[0]), np.cos(posture[1]), np.sin(posture[1])])
    x[0, :8] = np.r_[cue, pose]
    x[1, 8] = wind * 1.5
    x[2, :8] = np.r_[cue, pose]
    return x


class FrozenCable:
    """Sol's cable body with no plasticity: z <- rho*e^{i omega} z + tanh(C x)."""

    def __init__(self, rng, n=24):
        self.C = rng.normal(0, 0.45, (n, D))
        self.omega = rng.uniform(-1.35, 1.35, n)
        self.rho = rng.uniform(0.65, 0.89, n)
        self.B0 = rng.normal(0, 0.045, (2, n))  # drawn in Sol's order

    def state(self, xs):
        q = self.rho * np.exp(1j * self.omega)
        z = np.zeros(len(self.omega), complex)
        for x in xs:
            z = q * z + np.tanh(self.C @ x)
        return z.real

    def count(self):
        return self.C.size + self.omega.size + self.rho.size


class SharedReadout:
    name = "shared"

    def __init__(self, rng, n=24):
        self.body = FrozenCable(rng, n)
        self.B = self.body.B0.copy()

    def forward(self, xs):
        h = self.body.state(xs)
        return self.B @ h, h

    def learn(self, trace, g):
        h = trace
        self.B += 0.35 * np.outer(g, h) / (0.25 + h @ h)
        self.B = np.clip(self.B, -2.5, 2.5)

    def count(self):
        return self.body.count() + self.B.size


class LocalReadout:
    name = "local"

    def __init__(self, rng, n=24, routes=16, active=2, beta=2.5, lr_tune=0.018, cue_blind=False):
        self.body = FrozenCable(rng, n)
        self.active = active
        self.beta = beta
        self.lr_tune = lr_tune
        self.cue_blind = cue_blind
        if cue_blind:
            self.name = "local_cueblind"
        # every route starts with the same readout, so untrained behaviour equals 'shared'
        self.B = np.repeat(self.body.B0[None], routes, axis=0)
        # tunings over the final frame [cue(4), pose(4)]; no cue preference at start
        tune_rng = np.random.default_rng(rng.integers(1 << 31))
        p = tune_rng.uniform([-1.2, 0.3], [1.2, 2.4], size=(routes, 2))
        self.T = np.zeros((routes, 8))
        self.T[:, 4:] = np.c_[np.cos(p[:, 0]), np.sin(p[:, 0]), np.cos(p[:, 1]), np.sin(p[:, 1])]

    def route(self, xs):
        x = xs[-1, :8]
        diff = x[None] - self.T
        if self.cue_blind:
            diff = diff[:, 4:]
        s = -self.beta * np.sum(diff**2, axis=1)
        idx = np.argpartition(-s, self.active - 1)[: self.active]
        a = np.exp(s[idx] - s[idx].max())
        return idx, a / a.sum(), x

    def forward(self, xs):
        h = self.body.state(xs)
        idx, a, x = self.route(xs)
        y = np.einsum("k,kij,j->i", a, self.B[idx], h)
        return y, (h, idx, a, x)

    def learn(self, trace, g):
        h, idx, a, x = trace
        for k, w in zip(idx, a):
            self.B[k] += 0.35 * w * np.outer(g, h) / (0.25 + h @ h)
            self.T[k] += self.lr_tune * w * (x - self.T[k])
        self.B = np.clip(self.B, -2.5, 2.5)

    def count(self):
        return self.body.count() + self.B.size + self.T.size


def make(name, seed):
    rng = np.random.default_rng(seed)
    if name == "shared":
        return SharedReadout(rng)
    if name == "shared_big":  # readout parameter count matched to 'local'
        m = SharedReadout(rng, n=24 * 16)
        m.name = "shared_big"
        return m
    if name == "local":
        return LocalReadout(rng)
    if name == "local_cueblind":
        return LocalReadout(rng, cue_blind=True)
    raise ValueError(name)


def probe(model, world, posture, task, wind, rng):
    action, trace = model.forward(frames(posture, task, wind))
    d = rng.choice([-1.0, 1.0], size=2)
    plus = world.sense(action + SIGMA * d, task, wind, posture)[0]
    minus = world.sense(action - SIGMA * d, task, wind, posture)[0]
    g = (plus - minus) / (2 * SIGMA) * d
    model.learn(trace, g)


def evaluate(model, world, seed, postures=80):
    rng = np.random.default_rng(seed)
    rows = []
    for task in range(N_TASKS):
        dist, hit = [], []
        for _ in range(postures):
            posture = rng.uniform([-1.2, 0.3], [1.2, 2.4])
            for w in (1, -1):
                y = model.forward(frames(posture, task, w))[0]
                _, h, d = world.sense(y, task, w, posture)
                dist.append(d)
                hit.append(h)
        rows.append({"task": TASKS[task], "distance": float(np.mean(dist)), "contact": float(np.mean(hit))})
    return rows


def experiment(name, seed, world_mode, episodes=650, evaluation=80):
    rng = np.random.default_rng(9001 + seed)  # Sol's training stream
    model = make(name, 1701 + seed)  # Sol's model seed
    world = World(world_mode)
    phases = [{"after": "untrained", "tasks": evaluate(model, world, 3300 + seed, evaluation)}]
    for task in range(N_TASKS):
        for _ in range(episodes):
            posture = rng.uniform([-1.2, 0.3], [1.2, 2.4])
            wind = int(rng.choice([-1, 1]))
            probe(model, world, posture, task, wind, rng)
        phases.append({"after": TASKS[task], "tasks": evaluate(model, world, 3300 + seed, evaluation)})
    return {"name": name, "seed": seed, "world": world_mode, "parameters": int(model.count()), "phases": phases}
