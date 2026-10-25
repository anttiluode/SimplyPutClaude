"""Reach controllers. All map (posture, context cue) -> joint command dtheta.

Every controller learns from the same signal: after fumbling to contact, the
posture that worked is known, so the reach that would have worked is
    dtheta_target = theta_final - theta_start.
Learning turns the fumble into a reflex (search distilled into a direct reach).

Controllers
-----------
PlainMLP      one shared tanh network; everything shares every weight.
GatedMLP      the same network, but each context switches on a fixed random
              half of the hidden units (context-dependent gating, Masse et al.
              2018). The gate is HAND-ASSIGNED from the cue's identity.
RouteNet      many branches ("routes"). Each branch has a tuning (what input it
              resonates with) and its own small linear "cable function". The
              input picks its routes by resonance; only the routes that carried
              the signal are updated, and their tuning drifts toward the
              signals they carry. Nobody assigns routes to contexts: the cue is
              just part of the input.
RouteNet(cue_blind=True)
              ablation: resonance ignores the cue (posture only), so all
              contexts share the same routes. Tests whether separation comes
              from routing.

Prior art for RouteNet, stated plainly: it is a normalized radial-basis
mixture of local linear experts with competitive (winner-take-most) tuning,
in the family of locally weighted learning, mixture of experts and RBF
networks. What is being tested is whether experience-assigned routing alone
protects old skills, compared with an unprotected net and a hand-gated net.
"""

from __future__ import annotations

import numpy as np

N_CONTEXTS = 4
CUE_SCALE = 1.5
POSTURE_DIMS = 4


def features(theta: np.ndarray, context: int) -> np.ndarray:
    cue = np.zeros(N_CONTEXTS)
    cue[context] = CUE_SCALE
    return np.concatenate(
        [np.cos(theta[:1]), np.sin(theta[:1]), np.cos(theta[1:]), np.sin(theta[1:]), cue]
    )


IN_DIM = POSTURE_DIMS + N_CONTEXTS


class PlainMLP:
    name = "plain"

    def __init__(self, rng: np.random.Generator, hidden: int = 64, lr: float = 0.05):
        self.lr = lr
        self.W1 = rng.normal(0, 1 / np.sqrt(IN_DIM), (hidden, IN_DIM))
        self.b1 = np.zeros(hidden)
        self.W2 = rng.normal(0, 1 / np.sqrt(hidden), (2, hidden)) * 0.1
        self.b2 = np.zeros(2)

    def _mask(self, context: int):
        return None

    def n_params(self) -> int:
        return self.W1.size + self.b1.size + self.W2.size + self.b2.size

    def predict(self, x: np.ndarray, context: int) -> np.ndarray:
        h = np.tanh(self.W1 @ x + self.b1)
        m = self._mask(context)
        if m is not None:
            h = h * m
        return self.W2 @ h + self.b2

    def learn(self, x: np.ndarray, context: int, target: np.ndarray) -> None:
        pre = self.W1 @ x + self.b1
        h = np.tanh(pre)
        m = self._mask(context)
        hm = h if m is None else h * m
        y = self.W2 @ hm + self.b2
        err = y - target  # dL/dy for L = 0.5|y-t|^2
        gW2 = np.outer(err, hm)
        gh = self.W2.T @ err
        if m is not None:
            gh = gh * m
        gpre = gh * (1 - h**2)
        self.W2 -= self.lr * gW2
        self.b2 -= self.lr * err
        self.W1 -= self.lr * np.outer(gpre, x)
        self.b1 -= self.lr * gpre


class GatedMLP(PlainMLP):
    name = "gated"

    def __init__(self, rng: np.random.Generator, hidden: int = 64, lr: float = 0.05, keep: float = 0.5):
        super().__init__(rng, hidden=hidden, lr=lr)
        self.masks = (rng.random((N_CONTEXTS, hidden)) < keep).astype(float)

    def _mask(self, context: int):
        return self.masks[context]


class RouteNet:
    name = "route"

    def __init__(
        self,
        rng: np.random.Generator,
        branches: int = 32,
        active: int = 4,
        beta: float = 4.0,
        lr: float = 0.3,
        lr_tune: float = 0.05,
        cue_blind: bool = False,
    ):
        self.lr = lr
        self.lr_tune = lr_tune
        self.active = active
        self.beta = beta
        self.cue_blind = cue_blind
        if cue_blind:
            self.name = "route_cueblind"
        # Tunings start spread over posture space with NO cue preference.
        from .arm import random_postures

        postures = random_postures(rng, branches)
        self.C = np.zeros((branches, IN_DIM))
        for k, th in enumerate(postures):
            self.C[k, :POSTURE_DIMS] = features(th, 0)[:POSTURE_DIMS]
        self.W = np.zeros((branches, 2, IN_DIM + 1))  # cable function per route
        self.usage = np.zeros(branches)

    def n_params(self) -> int:
        return self.C.size + self.W.size

    def _route(self, x: np.ndarray):
        diff = x[None, :] - self.C
        if self.cue_blind:
            diff = diff[:, :POSTURE_DIMS]
        r = -self.beta * np.sum(diff**2, axis=1)
        if self.active >= len(r):
            idx = np.arange(len(r))
        else:
            idx = np.argpartition(-r, self.active)[: self.active]
        z = r[idx] - r[idx].max()
        a = np.exp(z)
        a /= a.sum()
        return idx, a

    def routes(self, x: np.ndarray) -> np.ndarray:
        """Indices of the routes this input travels through."""
        return np.sort(self._route(x)[0])

    def predict(self, x: np.ndarray, context: int) -> np.ndarray:
        idx, a = self._route(x)
        phi = np.append(x, 1.0)
        return np.einsum("k,kij,j->i", a, self.W[idx], phi)

    def learn(self, x: np.ndarray, context: int, target: np.ndarray) -> None:
        idx, a = self._route(x)
        phi = np.append(x, 1.0)
        norm = phi @ phi
        for k, ak in zip(idx, a):
            yk = self.W[k] @ phi
            # local: each route that carried the signal fits its own cable function
            self.W[k] += self.lr * ak * np.outer(target - yk, phi) / norm
            # its tuning drifts toward what it carries
            self.C[k] += self.lr_tune * ak * (x - self.C[k])
            self.usage[k] += ak


def make(name: str, rng: np.random.Generator, **kw):
    if name == "plain":
        return PlainMLP(rng, **kw)
    if name == "gated":
        return GatedMLP(rng, **kw)
    if name == "route":
        return RouteNet(rng, **kw)
    if name == "route_cueblind":
        return RouteNet(rng, cue_blind=True, **kw)
    raise ValueError(name)
