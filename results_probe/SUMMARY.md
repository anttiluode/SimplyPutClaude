# Part 2 results: learning from scalar probes, shared vs local readouts

First-reach distance to the target (lower is better; contact at 0.12). 5 seeds, 650 probe episodes per task, 80 test postures x 2 winds.

**Reproduction of SimplyPutSol v1 (frozen cable, seeds 0-2):** measured {'bell_learned': 0.0288, 'bell_final': 0.7139, 'final': 0.3949}, published {'bell_learned': 0.029, 'bell_final': 0.714, 'final': 0.395} -> PASS

## World: anchored

| Readout | params | bell just learned | bell after 3 more | final, 4 tasks | final contact |
|---|---:|---:|---:|---:|---:|
| shared | 312 | 0.038 | 0.649 | 0.361 +/- 0.046 | 30% |
| shared_big | 4992 | 0.071 | 0.510 | 0.397 +/- 0.015 | 26% |
| local | 1160 | 0.054 | 0.056 | 0.069 +/- 0.008 | 84% |
| local_cueblind | 1160 | 0.120 | 0.564 | 0.386 +/- 0.022 | 26% |

Bell distance after each phase:

| Readout | untrained | bell | switch | light | horn |
|---|---:|---:|---:|---:|---:|
| shared | 1.300 | 0.038 | 0.652 | 0.658 | 0.649 |
| shared_big | 1.108 | 0.071 | 0.359 | 0.391 | 0.510 |
| local | 1.300 | 0.054 | 0.056 | 0.067 | 0.056 |
| local_cueblind | 1.300 | 0.120 | 0.863 | 0.848 | 0.564 |

Rules:

- P1_learning_survives: PASS
- P2_retention: PASS
- P3_beats_shared: PASS
- P4_cueblind_fails_retention: PASS
- P5_beats_param_matched_shared: PASS

## World: relative

| Readout | params | bell just learned | bell after 3 more | final, 4 tasks | final contact |
|---|---:|---:|---:|---:|---:|
| shared | 312 | 0.147 | 1.048 | 0.576 +/- 0.085 | 13% |
| shared_big | 4992 | 0.134 | 0.624 | 0.486 +/- 0.066 | 12% |
| local | 1160 | 0.219 | 0.266 | 0.199 +/- 0.010 | 36% |
| local_cueblind | 1160 | 0.333 | 0.829 | 0.499 +/- 0.032 | 12% |

Bell distance after each phase:

| Readout | untrained | bell | switch | light | horn |
|---|---:|---:|---:|---:|---:|
| shared | 1.364 | 0.147 | 0.884 | 0.843 | 1.048 |
| shared_big | 1.289 | 0.134 | 0.546 | 0.413 | 0.624 |
| local | 1.364 | 0.219 | 0.219 | 0.221 | 0.266 |
| local_cueblind | 1.364 | 0.333 | 0.941 | 0.916 | 0.829 |

Rules:

- P1_learning_survives: PASS
- P2_retention: PASS
- P3_beats_shared: PASS
- P4_cueblind_fails_retention: PASS
- P5_beats_param_matched_shared: PASS

Runtime 25.5 s.
