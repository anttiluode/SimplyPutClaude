# Results

Corrections needed to touch the target (lower is better; 60 = gave up). Mean over 10 seeds, 200 test postures each.

## After learning all four, in order bell → switch → light → horn

| Controller | params | lr | bell | switch | light | horn | average | bell kept (final/just learned) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| plain | 706 | 0.003 | 3.42 | 3.13 | 3.25 | 1.95 | **2.94** ± 0.13 | 1.96 ± 0.57 |
| gated | 706 | 0.01 | 4.50 | 3.69 | 3.47 | 1.88 | **3.39** ± 0.14 | 2.60 ± 0.34 |
| route | 832 | 1.0 | 0.75 | 0.65 | 0.48 | 0.85 | **0.68** ± 0.04 | 0.98 ± 0.05 |
| route_cueblind | 832 | 1.5 | 4.05 | 4.51 | 5.00 | 0.58 | **3.53** ± 0.01 | 2.65 ± 0.25 |
| plain, all tasks mixed (reference) | | | 1.71 | 1.96 | 1.41 | 2.00 | 1.77 ± 0.03 | — |
| no learning | | | 5.79 | 6.48 | 5.43 | 6.80 | 6.13 | — |

## Right after learning each task (before later tasks)

| Controller | bell | switch | light | horn |
|---|---:|---:|---:|---:|
| plain | 1.76 | 1.87 | 1.37 | 1.95 |
| gated | 1.74 | 1.93 | 1.45 | 1.88 |
| route | 0.77 | 0.66 | 0.48 | 0.85 |
| route_cueblind | 1.55 | 1.09 | 0.57 | 0.58 |

## Bell over time (corrections on the bell after each phase)

| Controller | before | after bell | after switch | after light | after horn |
|---|---:|---:|---:|---:|---:|
| plain | 5.79 | 1.76 | 3.65 | 4.14 | 3.42 |
| gated | 5.80 | 1.74 | 3.69 | 4.45 | 4.50 |
| route | 5.80 | 0.77 | 0.77 | 0.77 | 0.75 |
| route_cueblind | 5.80 | 1.55 | 5.11 | 5.23 | 4.05 |

## Capacity sweep (RouteNet, 5 seeds)

| routes | bell just learned | bell at the end | average at the end |
|---:|---:|---:|---:|
| 4 | 0.58 | 3.69 | 2.66 |
| 8 | 0.43 | 2.93 | 2.59 |
| 16 | 0.43 | 0.74 | 0.79 |
| 32 | 0.71 | 0.70 | 0.65 |
| 64 | 0.62 | 0.62 | 0.66 |

## Which routes each task uses (Jaccard overlap, RouteNet, 5 seeds)

**cue reaches the routing**

| | bell | switch | light | horn |
|---|---:|---:|---:|---:|
| bell | 1.00 | 0.31 | 0.28 | 0.34 |
| switch | 0.31 | 1.00 | 0.32 | 0.39 |
| light | 0.28 | 0.32 | 1.00 | 0.32 |
| horn | 0.34 | 0.39 | 0.32 | 1.00 |

**cue-blind routing**

| | bell | switch | light | horn |
|---|---:|---:|---:|---:|
| bell | 1.00 | 1.00 | 1.00 | 1.00 |
| switch | 1.00 | 1.00 | 1.00 | 1.00 |
| light | 1.00 | 1.00 | 1.00 | 1.00 |
| horn | 1.00 | 1.00 | 1.00 | 1.00 |

## Learning-rate selection (bell only, seed 999)

- plain: chose 0.003 from 0.001→2.12, 0.003→1.68, 0.01→1.73, 0.03→1.75, 0.1→2.27
- gated: chose 0.01 from 0.001→3.29, 0.003→1.77, 0.01→1.68, 0.03→1.73, 0.1→1.75
- route: chose 1.0 from 0.1→2.56, 0.3→1.32, 1.0→0.24, 1.5→0.35
- route_cueblind: chose 1.5 from 0.1→2.48, 0.3→2.00, 1.0→1.68, 1.5→1.51

Runtime 133.1 s.
