import numpy as np

from simplyput.probe import LocalReadout, World, frames, make, probe


def test_untrained_local_equals_shared():
    xs = frames(np.array([0.3, 1.1]), 2, 1)
    a = make("shared", 1701).forward(xs)[0]
    b = make("local", 1701).forward(xs)[0]
    assert np.allclose(a, b)


def test_local_update_touches_only_active_routes():
    m = LocalReadout(np.random.default_rng(3))
    xs = frames(np.array([0.2, 1.0]), 0, -1)
    _, trace = m.forward(xs)
    used = set(trace[1].tolist())
    B, T = m.B.copy(), m.T.copy()
    m.learn(trace, np.array([0.3, -0.2]))
    changed = {k for k in range(len(B)) if not (np.array_equal(B[k], m.B[k]) and np.array_equal(T[k], m.T[k]))}
    assert changed == used


def test_body_remembers_wind_but_final_frame_does_not_show_it():
    a, b = frames(np.array([0.2, 1.0]), 0, 1), frames(np.array([0.2, 1.0]), 0, -1)
    assert np.array_equal(a[-1], b[-1])
    m = make("shared", 5)
    assert not np.allclose(m.forward(a)[0], m.forward(b)[0])


def test_anchored_world_ignores_posture_relative_world_does_not():
    action = np.array([0.1, -0.1])
    p1, p2 = np.array([-0.5, 0.8]), np.array([0.7, 2.0])
    A, R = World("anchored"), World("relative")
    assert A.sense(action, 0, 1, p1) == A.sense(action, 0, 1, p2)
    assert R.sense(action, 0, 1, p1) != R.sense(action, 0, 1, p2)


def test_probe_learning_reduces_bell_distance():
    world = World("anchored")
    rng = np.random.default_rng(0)
    m = make("local", 7)
    xs_post = [(rng.uniform([-1.2, 0.3], [1.2, 2.4]), w) for w in (1, -1) for _ in range(20)]
    before = np.mean([world.sense(m.forward(frames(p, 0, w))[0], 0, w, p)[2] for p, w in xs_post])
    for _ in range(400):
        probe(m, world, rng.uniform([-1.2, 0.3], [1.2, 2.4]), 0, int(rng.choice([-1, 1])), rng)
    after = np.mean([world.sense(m.forward(frames(p, 0, w))[0], 0, w, p)[2] for p, w in xs_post])
    assert after < 0.3 * before
