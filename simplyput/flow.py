"""Part 3: the flow loop.

    probe -> return -> read (multiscale) -> guess -> told the answer
          -> write into that object's own memory route -> choose next probe

The world hides one of 8 objects. Probing band b makes the object return two
Gabor bursts at two bands, in some order. Objects come in twin pairs: twins
return the same two bands on every probe, but in reversed order on a few
bands (different bands for each pair). So telling twins apart needs (a) a
reader that sees ORDER and (b) a probe policy that finds the bands where
order differs.

Readers
  R0  total energy
  R1  first-order scattering, globally averaged (a spectrum)
  R2  first + second-order scattering, globally averaged
  R3  first order + lagged cross-band envelope asymmetry
      (a minimal joint time-frequency statistic: does band i's energy come
      before or after band j's?)

Memory: one running-mean template per (object, band): local writes only.

Policies
  random      random distinct bands
  fixed       the best fixed pair of bands, chosen with oracle knowledge
  curious     closed loop: probe where candidate memories disagree most;
              memory never written counts as maximal disagreement
  greedy      closed loop without curiosity: unwritten memory counts as zero
"""

from __future__ import annotations

import itertools

import numpy as np

T = 512
B = 8
FREQS = 0.04 * 1.35 ** np.arange(B)       # cycles/sample, log spaced
BURST_SIGMA = 20.0
T1, T2 = 140, 340
JITTER_T, JITTER_A, NOISE = 10, 0.15, 0.25
N_GROUPS = 4
N_OBJ = 2 * N_GROUPS
PAIR_POOL = 4
FLIPS_PER_GROUP = 3
POOL_PER_CELL = 40

# ---------------------------------------------------------------- world


def make_world(seed):
    """Return per (object, probe band) the two echo bands in order."""
    rng = np.random.default_rng(seed)
    pairs = [tuple(rng.choice(B, 2, replace=False)) for _ in range(PAIR_POOL * 3)]
    pairs = list(dict.fromkeys(tuple(sorted(p)) for p in pairs))[:PAIR_POOL]
    echo = np.zeros((N_OBJ, B, 2), dtype=int)
    flips = np.zeros((N_GROUPS, B), dtype=bool)
    for g in range(N_GROUPS):
        flips[g, rng.choice(B, FLIPS_PER_GROUP, replace=False)] = True
        for b in range(B):
            a, c = pairs[rng.integers(len(pairs))]
            if rng.random() < 0.5:
                a, c = c, a
            echo[2 * g, b] = (a, c)
            echo[2 * g + 1, b] = (c, a) if flips[g, b] else (a, c)
    return echo, flips


def gabor(f, t0, amp):
    t = np.arange(T)
    return amp * np.exp(-0.5 * ((t - t0) / BURST_SIGMA) ** 2) * np.cos(2 * np.pi * f * t)


def sense(echo, obj, band, rng):
    first, second = echo[obj, band]
    x = gabor(FREQS[first], T1 + rng.integers(-JITTER_T, JITTER_T + 1), 1 + JITTER_A * rng.normal())
    x += gabor(FREQS[second], T2 + rng.integers(-JITTER_T, JITTER_T + 1), 1 + JITTER_A * rng.normal())
    return x + NOISE * rng.normal(size=T)


# ---------------------------------------------------------------- readers

_freq = np.fft.fftfreq(T)


def _morlet_bank(centers, q=1.0):
    """Analytic Gabor filters in the frequency domain, bandwidth proportional to centre."""
    bank = []
    for fc in centers:
        bw = fc / (2.5 * q)
        bank.append(np.exp(-0.5 * ((_freq - fc) / bw) ** 2) * (_freq > 0))
    return np.array(bank)


BANK1 = _morlet_bank(FREQS)
MOD_FREQS = np.array([0.004, 0.008, 0.016, 0.032])
BANK2 = np.array([np.exp(-0.5 * ((np.abs(_freq) - fc) / (fc / 2)) ** 2) for fc in MOD_FREQS])
PAIRS = list(itertools.combinations(range(B), 2))


def envelopes(x):
    X = np.fft.fft(x)
    return np.abs(np.fft.ifft(X[None] * BANK1, axis=1))  # B x T


def read(x, reader):
    if reader == "R0":
        return np.array([np.log1p(np.sum(x**2))])
    E = envelopes(x)
    s1 = np.log1p(E.mean(1) * 10)
    if reader == "R1":
        return s1
    if reader == "R2":
        F = np.fft.fft(E, axis=1)
        s2 = np.abs(np.fft.ifft(F[:, None, :] * BANK2[None], axis=2)).mean(2)  # B x 4
        return np.r_[s1, np.log1p(s2.ravel() * 10)]
    if reader == "R3":
        En = E - E.mean(1, keepdims=True)
        F = np.fft.fft(En, n=2 * T, axis=1)
        asym = []
        for i, j in PAIRS:
            xc = np.real(np.fft.ifft(np.conj(F[i]) * F[j]))  # xc[k] = sum_t e_i(t) e_j(t+k)
            later = xc[1:T // 2].sum()          # j after i
            earlier = xc[-(T // 2) + 1:].sum()  # j before i
            norm = np.sqrt(np.sum(En[i] ** 2) * np.sum(En[j] ** 2)) * T / 4 + 1e-9
            asym.append((later - earlier) / norm)
        return np.r_[s1, np.array(asym) * 5]
    raise ValueError(reader)


def feature_pools(echo, reader, seed):
    """Pre-drawn noisy readings per (object, band): separate pools for learning and testing."""
    rng = np.random.default_rng(seed + 12345)
    out = {}
    for split in ("learn", "test"):
        out[split] = np.array([[[read(sense(echo, k, b, rng), reader) for _ in range(POOL_PER_CELL)]
                                for b in range(B)] for k in range(N_OBJ)])
    return out  # [obj, band, sample, dim]


# ---------------------------------------------------------------- memory + inference


class Memory:
    """One template per (object, band). Writes touch only that cell."""

    def __init__(self, dim):
        self.mu = np.zeros((N_OBJ, B, dim))
        self.n = np.zeros((N_OBJ, B))
        self.resid = np.zeros(dim)
        self.n_resid = 0
        self.all_sum = np.zeros((B, dim))
        self.all_sq = np.zeros((B, dim))
        self.all_n = np.zeros(B)

    def var(self):
        # pooled within-cell variance, with a weak prior so early guesses aren't overconfident
        return (self.resid + 5 * 0.05) / (self.n_resid + 5) + 0.01

    def broad(self, b):
        n = max(self.all_n[b], 1)
        m = self.all_sum[b] / n
        v = np.maximum(self.all_sq[b] / n - m**2, 0) + self.var() + 0.5
        return m, v

    def loglik(self, b, f):
        """log p(f | object k, band b) for every k; unwritten cells use a broad likelihood."""
        v = self.var()
        out = np.empty(N_OBJ)
        mb, vb = self.broad(b)
        for k in range(N_OBJ):
            if self.n[k, b] > 0:
                out[k] = -0.5 * np.sum((f - self.mu[k, b]) ** 2 / v + np.log(v))
            else:
                out[k] = -0.5 * np.sum((f - mb) ** 2 / vb + np.log(vb))
        return out

    def write(self, k, b, f):
        if self.n[k, b] > 0:
            self.resid += (f - self.mu[k, b]) ** 2
            self.n_resid += 1
        self.n[k, b] += 1
        self.mu[k, b] += (f - self.mu[k, b]) / self.n[k, b]
        self.all_sum[b] += f
        self.all_sq[b] += f**2
        self.all_n[b] += 1


def disagreement(mem, post, b, curious):
    v = mem.var()
    known = mem.n[:, b] > 0
    score = 0.0
    if known.any():
        w = post[known] / max(post[known].sum(), 1e-12)
        m = (w[:, None] * mem.mu[known, b]).sum(0)
        score += post[known].sum() * float((w * (((mem.mu[known, b] - m) ** 2) / v).sum(1)).sum())
    if curious and (~known).any():
        # an unwritten cell is expected to differ by about the spread of everything seen in this band
        _, vb = mem.broad(b)
        score += float(post[~known].sum()) * float(np.sum(vb / v))
    return score


def choose(policy, mem, post, used, rng, fixed_pair, step):
    free = [b for b in range(B) if b not in used]
    if policy == "random":
        return int(rng.choice(free))
    if policy == "fixed":
        return fixed_pair[step]
    scores = [disagreement(mem, post, b, curious=(policy == "curious")) for b in free]
    best = np.flatnonzero(np.isclose(scores, max(scores)))
    return free[int(rng.choice(best))]


def episode(mem, pools, split, obj, policy, rng, fixed_pair, probes=2, learn=True):
    logp = np.zeros(N_OBJ)
    used, readings = [], []
    for step in range(probes):
        post = np.exp(logp - logp.max())
        post /= post.sum()
        b = choose(policy, mem, post, used, rng, fixed_pair, step)
        f = pools[split][obj, b, rng.integers(POOL_PER_CELL)]
        logp += mem.loglik(b, f)
        used.append(b)
        readings.append((b, f))
    guess = int(np.argmax(logp + 1e-9 * rng.normal(size=N_OBJ)))
    if learn:
        for b, f in readings:
            mem.write(obj, b, f)
    return guess


def best_fixed_pair(echo, flips):
    """Oracle open-loop choice: the band pair separating the most objects (order included)."""
    best, best_n = None, -1
    for p in itertools.combinations(range(B), 2):
        sigs = {tuple(tuple(echo[k, b]) for b in p) for k in range(N_OBJ)}
        n = len(sigs)
        if n > best_n:
            best, best_n = p, n
    return list(best)


def life(seed, reader, policy, learn_episodes=600, test_episodes=400, pools=None):
    echo, flips = make_world(seed)
    if pools is None:
        pools = feature_pools(echo, reader, seed)
    mem = Memory(pools["learn"].shape[-1])
    rng = np.random.default_rng(seed + 999)
    fixed_pair = best_fixed_pair(echo, flips)
    curve = []
    for _ in range(learn_episodes):
        obj = int(rng.integers(N_OBJ))
        curve.append(episode(mem, pools, "learn", obj, policy, rng, fixed_pair) == obj)
    correct, twin_ok, twin_n = 0, 0, 0
    for _ in range(test_episodes):
        obj = int(rng.integers(N_OBJ))
        g = episode(mem, pools, "test", obj, policy, rng, fixed_pair, learn=False)
        correct += g == obj
        if g // 2 == obj // 2:
            twin_n += 1
            twin_ok += g == obj
    return {
        "test_acc": correct / test_episodes,
        "twin_acc": twin_ok / max(twin_n, 1),
        "pair_acc": twin_n / test_episodes,
        "curve": np.array(curve, float),
        "coverage": float((mem.n > 0).mean()),
        "fixed_pair": fixed_pair,
    }
