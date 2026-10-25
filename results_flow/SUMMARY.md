# Part 3 results: the flow loop

5 seeds; 8 hidden objects in 4 twin pairs; 2 probes per episode; 600 learning episodes, then 400 test episodes with no writing. Chance = 0.125.

## Test accuracy (identify the hidden object)

| Reader | random | fixed | curious | greedy |
|---|---:|---:|---:|---:|
| R0 | 0.132 +/- 0.008 | 0.117 +/- 0.010 | 0.137 +/- 0.008 | 0.122 +/- 0.010 |
| R1 | 0.439 +/- 0.010 | 0.447 +/- 0.036 | 0.496 +/- 0.015 | 0.415 +/- 0.053 |
| R2 | 0.450 +/- 0.013 | 0.418 +/- 0.037 | 0.493 +/- 0.021 | 0.469 +/- 0.015 |
| R3 | 0.757 +/- 0.017 | 0.966 +/- 0.021 | 0.873 +/- 0.037 | 0.741 +/- 0.082 |

## Twin accuracy (right object, given the right twin pair was found)

| Reader | random | fixed | curious | greedy |
|---|---:|---:|---:|---:|
| R0 | 0.494 | 0.487 | 0.568 | 0.506 |
| R1 | 0.491 | 0.501 | 0.497 | 0.491 |
| R2 | 0.499 | 0.470 | 0.493 | 0.470 |
| R3 | 0.802 | 0.999 | 0.948 | 0.793 |

## Memory coverage (fraction of object x band cells ever written)

| Reader | random | fixed | curious | greedy |
|---|---:|---:|---:|---:|
| R0 | 1.00 | 0.25 | 1.00 | 0.30 |
| R1 | 1.00 | 0.25 | 1.00 | 0.30 |
| R2 | 1.00 | 0.25 | 1.00 | 0.32 |
| R3 | 1.00 | 0.25 | 1.00 | 0.54 |

## Rules

- F1_order_needs_joint_reader: PASS
- F2_loop_beats_random: PASS
- F3_loop_beats_best_fixed_pair: FAIL
- F4_curiosity_needed: FAIL

Per-seed test accuracy, R3:

- random: 0.755, 0.772, 0.705, 0.810, 0.743
- fixed: 0.990, 0.890, 0.958, 1.000, 0.995
- curious: 0.868, 0.767, 0.870, 1.000, 0.860
- greedy: 0.510, 0.825, 0.585, 0.858, 0.927

Oracle fixed pairs per seed: [[0, 2], [0, 1], [1, 6], [0, 6], [4, 7]]

Runtime 65.5 s.
