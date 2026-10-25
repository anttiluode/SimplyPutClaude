# Decision rules, written before the main run

The prototype (seed 0, 800 reaches per task) had already been seen when these
were written. The main run uses 1000 reaches per task and seeds 0–9, and learning
rates are chosen on a separate bell-only run (seed 999) so forgetting cannot
influence them.

RouteNet "earns its place" in this toy only if all four hold:

1. **Retention.** Bell corrections at the end are at most 1.25× the bell
   corrections right after learning the bell (mean over seeds).
2. **Better than the shared controllers.** Average corrections over all four
   tasks at the end are lower than both the plain net and the hand-gated net
   by more than 2 standard errors.
3. **The routing is the cause.** The cue-blind ablation (same model, routing
   ignores the cue) fails rule 1.
4. **Mechanism check.** In the capacity sweep, retention gets worse as the
   number of routes shrinks toward 4, where every input uses every route.

Also reported regardless of outcome: how well each controller learns a task
right after practising it (a separate-routes design may learn each task worse),
and how far the plain net gets when tasks are mixed instead of sequential.

What passing would NOT show: that this is new. RouteNet belongs to a known
family (normalized RBF networks, mixtures of local linear experts, locally
weighted learning). Gating alone is also a weak bar; Masse et al. (2018)
combined gating with synaptic stabilization. Beating stronger continual-learning
baselines (EWC, synaptic intelligence, replay) has not been tested.

---

# Part 2: probe learning (written before running it)

Sol's SimplyPutSol v1 learned from scalar probes and forgot (bell 0.029 →
0.714 with frozen cables), while my v1 kept skills but used a shortcut teacher.
Part 2 keeps Sol's world, probes, frozen cable body and readout update rule
exactly, and changes only the readout: shared vs 16 route-local readouts (2
active, routing on the final frame, tunings start cue-neutral). Not tuned:
routes=16, active=2, beta=2.5 and the tuning rate are fixed here before any run.
5 seeds, 650 probe episodes per task, 80 test postures × 2 winds.

Run in two worlds: `anchored` (Sol's: action added to a fixed anchor posture)
and `relative` (action added to the current posture).

Rules, per world:

- **P0 reproduction.** In the anchored world, `shared` on Sol's seeds 0–2
  reproduces Sol's published frozen-cable numbers (bell learned 0.029, bell
  final 0.714, final four-task 0.395) to within 0.005.
- **P1 learning survives locality.** `local` learns the bell to a distance no
  worse than 2× `shared`'s just-learned distance (or ≤ 0.06).
- **P2 retention.** `local` bell final ≤ max(1.5 × bell just learned,
  just learned + 0.05).
- **P3 better overall.** `local` final four-task distance lower than `shared`
  by more than 2 standard errors.
- **P4 cause.** `local_cueblind` fails P2.
- **P5 not just more weights.** `local` final four-task distance lower than
  `shared_big` (readout parameters matched) by more than 2 standard errors.
