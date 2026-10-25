"""Two-joint planar arm, hidden targets, and the corrective (fumbling) loop.

The arm has a shoulder at the origin and two links. A posture is (theta1, theta2).
A reach is a one-shot joint command dtheta: the hand goes to posture theta + dtheta.

If the hand misses, the arm fumbles toward the target with small corrective
movements. Each correction uses the endpoint error the hand can feel near the
target (a 2-D touch/near-field error vector) and a damped least-squares step.
The number of corrections needed is the measure of skill: a learned reach
needs few or none.

Limitation, stated up front: the feedback is a 2-D error vector, not a bare
binary click. A click-only version would need search, and is left for later.
"""

from __future__ import annotations

import numpy as np

L1 = 1.0
L2 = 0.8

THETA1_RANGE = (-np.pi / 2, np.pi / 2)
THETA2_RANGE = (0.2, 2.6)  # elbow kept on one side: unique inverse kinematics

CONTACT_RADIUS = 0.05
CORRECTION_GAIN = 0.5
CORRECTION_DAMPING = 0.05
MAX_CORRECTIONS = 60


def hand(theta: np.ndarray) -> np.ndarray:
    """Hand position for posture theta (shape (..., 2))."""
    t1 = theta[..., 0]
    t12 = theta[..., 0] + theta[..., 1]
    x = L1 * np.cos(t1) + L2 * np.cos(t12)
    y = L1 * np.sin(t1) + L2 * np.sin(t12)
    return np.stack([x, y], axis=-1)


def jacobian(theta: np.ndarray) -> np.ndarray:
    """2x2 Jacobian d(hand)/d(theta) at a single posture."""
    t1, t2 = theta
    t12 = t1 + t2
    return np.array(
        [
            [-L1 * np.sin(t1) - L2 * np.sin(t12), -L2 * np.sin(t12)],
            [L1 * np.cos(t1) + L2 * np.cos(t12), L2 * np.cos(t12)],
        ]
    )


def clip_posture(theta: np.ndarray) -> np.ndarray:
    out = np.array(theta, dtype=float)
    out[..., 0] = np.clip(out[..., 0], -np.pi, np.pi)
    out[..., 1] = np.clip(out[..., 1], THETA2_RANGE[0], THETA2_RANGE[1])
    return out


def random_postures(rng: np.random.Generator, n: int) -> np.ndarray:
    t1 = rng.uniform(*THETA1_RANGE, size=n)
    t2 = rng.uniform(*THETA2_RANGE, size=n)
    return np.stack([t1, t2], axis=-1)


def correction_step(theta: np.ndarray, target: np.ndarray) -> np.ndarray:
    """One small corrective movement toward the felt target (damped least squares)."""
    e = target - hand(theta)
    J = jacobian(theta)
    step = J.T @ np.linalg.solve(J @ J.T + CORRECTION_DAMPING * np.eye(2), e)
    return clip_posture(theta + CORRECTION_GAIN * step)


def fumble(theta_after_reach: np.ndarray, target: np.ndarray):
    """Correct until contact. Returns (final posture, number of corrections, reached)."""
    theta = clip_posture(theta_after_reach)
    for n in range(MAX_CORRECTIONS + 1):
        if np.linalg.norm(target - hand(theta)) <= CONTACT_RADIUS:
            return theta, n, True
        if n == MAX_CORRECTIONS:
            break
        theta = correction_step(theta, target)
    return theta, MAX_CORRECTIONS, False


def target_from_posture(theta_star) -> np.ndarray:
    """Targets are defined by a reachable posture, so every target is reachable."""
    return hand(np.asarray(theta_star, dtype=float))


# Four hidden things, as postures (theta1, theta2). Bell first.
TASK_POSTURES = {
    "bell": (-1.10, 1.60),
    "switch": (0.90, 1.00),
    "light": (0.10, 2.20),
    "horn": (-0.40, 0.60),
}
TASK_ORDER = ["bell", "switch", "light", "horn"]
