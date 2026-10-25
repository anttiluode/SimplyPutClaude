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
