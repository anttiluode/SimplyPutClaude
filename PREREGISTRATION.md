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

---

# Part 3: the flow loop (written before running it)

The loop: probe a hidden object, read the return with a multiscale reader,
guess, get told the answer, write the reading into that object's own memory
route, choose the next probe from what has been learned. The question is
whether **closing the loop** (choosing probes from memory) beats the same parts
run open-loop, and which reader is needed to remember **order**.

World: 8 hidden objects in 4 twin pairs. A probe in band b returns two Gabor
bursts at two bands, in some order. Twins return the same pair of bands on
every probe, and differ only in the order of the two bursts, and only on some
bands (3 of 8, different per pair). Timing jitter, amplitude jitter and noise
on every return. 2 probes per episode, 600 learning episodes, then 400 test
episodes with no writing. 5 seeds.

Readers: R0 total energy; R1 first-order scattering, globally averaged (a
spectrum); R2 first + second-order scattering, globally averaged; R3 first
order + lagged cross-band envelope asymmetry (a minimal joint time-frequency
statistic).

Policies: random bands; best fixed pair of bands (chosen with full knowledge of
the world, an open-loop upper bound); closed loop (choose the band where the
candidate objects' memories disagree most, unknown memory counted as maximal
disagreement); greedy closed loop (same, but unknown memory counts as zero:
no curiosity).

Prediction change, stated before running: in the conversation I predicted that
second-order scattering would see ordering. On reflection it should not:
globally averaged scattering is invariant to when each band's energy occurs,
and swapping the order of two bands only moves each band's bump in time. Order
needs relative timing between bands. Rules:

- **F1 order needs a joint time-frequency reader.** With the closed-loop policy,
  twin accuracy (correct given the right pair was found) is ≤ 0.65 for R1 and
  R2, and R3's twin accuracy exceeds R2's by more than 0.25.
- **F2 closing the loop helps.** With R3, closed-loop test accuracy beats random
  by more than 2 standard errors.
- **F3 it beats the best open-loop choice.** With R3, closed-loop test accuracy
  beats the best fixed pair (chosen with oracle knowledge) by more than 2 SE.
- **F4 curiosity is necessary.** With R3, the greedy loop (no credit for
  unknown memory) ends below the curious loop by more than 2 SE.

Known prior art: active learning / optimal experimental design (query by
disagreement), nearest-class-mean memory, joint time-frequency scattering
(Andén, Lostanlen, Mallat). Passing would not show novelty.
