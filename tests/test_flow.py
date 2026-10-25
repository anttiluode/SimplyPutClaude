import numpy as np

from simplyput.flow import B, N_OBJ, Memory, best_fixed_pair, make_world, read, sense


def test_twins_share_bands_and_differ_only_in_order_on_flip_bands():
    echo, flips = make_world(0)
    for g in range(N_OBJ // 2):
        a, b = echo[2 * g], echo[2 * g + 1]
        for band in range(B):
            assert sorted(a[band]) == sorted(b[band])
            assert (tuple(a[band]) != tuple(b[band])) == flips[g, band]


def _mean_reading(echo, obj, band, reader, n=12, seed=0):
    rng = np.random.default_rng(seed)
    return np.mean([read(sense(echo, obj, band, rng), reader) for _ in range(n)], axis=0)


def test_spectrum_reader_cannot_see_order_but_joint_reader_can():
    echo, flips = make_world(0)
    g = 0
    band = int(np.flatnonzero(flips[g])[0])
    for reader, should_differ in (("R1", False), ("R3", True)):
        a = _mean_reading(echo, 2 * g, band, reader)
        b = _mean_reading(echo, 2 * g + 1, band, reader, seed=1)
        noise = np.linalg.norm(_mean_reading(echo, 2 * g, band, reader, seed=2) - a)
        gap = np.linalg.norm(a - b)
        assert (gap > 4 * noise) == should_differ, (reader, gap, noise)


def test_memory_write_is_local():
    m = Memory(5)
    before = m.mu.copy(), m.n.copy()
    m.write(3, 2, np.ones(5))
    changed = np.argwhere(np.any(m.mu != before[0], axis=2) | (m.n != before[1]))
    assert changed.tolist() == [[3, 2]]


def test_best_fixed_pair_is_two_distinct_bands():
    echo, flips = make_world(1)
    p = best_fixed_pair(echo, flips)
    assert len(p) == 2 and p[0] != p[1]
