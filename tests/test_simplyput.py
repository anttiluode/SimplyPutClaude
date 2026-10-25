import numpy as np

from simplyput import arm
from simplyput.models import IN_DIM, RouteNet, features, make
from simplyput.protocol import attempt, evaluate, practice, task_target


def test_jacobian_matches_finite_difference():
    rng = np.random.default_rng(0)
    for th in arm.random_postures(rng, 20):
        J = arm.jacobian(th)
        eps = 1e-6
        num = np.stack(
            [(arm.hand(th + eps * e) - arm.hand(th - eps * e)) / (2 * eps) for e in np.eye(2)], axis=1
        )
        assert np.allclose(J, num, atol=1e-6)


def test_fumble_reaches_every_target_from_random_postures():
    rng = np.random.default_rng(1)
    for task in arm.TASK_ORDER:
        target = task_target(task)
        for th in arm.random_postures(rng, 50):
            final, n, ok = arm.fumble(th, target)
            assert ok, (task, th)
            assert np.linalg.norm(arm.hand(final) - target) <= arm.CONTACT_RADIUS


def test_perfect_reach_needs_zero_corrections():
    th0 = np.array([0.3, 1.2])
    star = np.array(arm.TASK_POSTURES["bell"])
    _, n, ok = arm.fumble(th0 + (star - th0), task_target("bell"))
    assert ok and n == 0


def test_targets_are_distinct():
    pts = np.stack([task_target(t) for t in arm.TASK_ORDER])
    d = np.linalg.norm(pts[:, None] - pts[None], axis=-1)
    assert d[~np.eye(4, dtype=bool)].min() > 0.3


def test_every_controller_learns_the_bell():
    for name in ["plain", "gated", "route", "route_cueblind"]:
        rng = np.random.default_rng(2)
        m = make(name, rng)
        postures = arm.random_postures(np.random.default_rng(3), 60)
        before = evaluate(m, postures, ["bell"])["bell"]["corrections"]
        practice(m, rng, "bell", 600)
        after = evaluate(m, postures, ["bell"])["bell"]["corrections"]
        assert after < 0.5 * before, (name, before, after)


def test_evaluation_does_not_learn():
    rng = np.random.default_rng(4)
    m = make("plain", rng)
    W = m.W1.copy()
    evaluate(m, arm.random_postures(rng, 10))
    assert np.array_equal(W, m.W1)


def test_route_update_is_local():
    rng = np.random.default_rng(5)
    m = RouteNet(rng)
    x = features(np.array([0.2, 1.0]), 0)
    used = set(m.routes(x).tolist())
    W, C = m.W.copy(), m.C.copy()
    m.learn(x, 0, np.array([0.1, -0.2]))
    changed = {k for k in range(len(W)) if not np.array_equal(W[k], m.W[k]) or not np.array_equal(C[k], m.C[k])}
    assert changed == used


def test_route_tunings_start_without_cue_preference():
    m = RouteNet(np.random.default_rng(6))
    assert np.all(m.C[:, IN_DIM - 4:] == 0)


def test_gating_masks_are_fixed_per_context():
    m = make("gated", np.random.default_rng(7))
    masks = m.masks.copy()
    practice(m, np.random.default_rng(8), "bell", 50)
    assert np.array_equal(masks, m.masks)
    assert not np.array_equal(m.masks[0], m.masks[1])


def test_attempt_is_deterministic():
    th = np.array([0.1, 1.5])
    a = attempt(make("route", np.random.default_rng(9)), th, "bell", 0, learn=False)
    b = attempt(make("route", np.random.default_rng(9)), th, "bell", 0, learn=False)
    assert a == b
