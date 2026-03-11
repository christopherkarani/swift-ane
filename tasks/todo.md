# TODO

- [x] Preserve the failed local real-artifact hardware route only as docs/results/memory; keep the implementation clean.
- [x] Re-run the exact synthetic `echo` harness with repeated fresh-process medians and verify parity plus `committed_exact_tokens/pass > 1`.
- [x] Package the synthetic `echo` result as the only publishable `>= 3x` claim surface on this branch, with explicit separation from the weaker `recurrent-checkpoint` route.
- [x] If the refreshed `echo` median drops below `3x`, try exactly one bounded performance hypothesis on the winning exact path and remeasure.
- [x] Update docs, lessons, Wax notes, handoff, and review with the final claim scope and every rejected avenue.

# Review

- Current strongest honest positive result on this branch is still the exact synthetic `echo` same-session harness at about `3.6986x` over matched CoreML, with parity `match`, `committed_exact_tokens/pass = 2`, and `accepted_future_tokens/pass = 1`.
- The local real-artifact avenue is now a strong negative result, not a publishable claim: the ANE recurrent hardware path outputs all-zero tokens on focused one-hot seams and on the local bigram artifact, while the direct ANE local-artifact spike also collapses to zeros and lands at about `0.98x` vs CoreML.
- Two independent fresh-process `7`-repeat reruns of the explicit synthetic `echo` harness now re-confirm a publishable `>= 3x` scope on this branch:
  - `3.5383781787824304x` at `1.6220833333333333 ms/token` vs matched CoreML `5.805169270833332 ms/token`
  - `3.7377633193960738x` at `1.5657968750000002 ms/token` vs matched CoreML `5.730479166666667 ms/token`
  - both reruns kept `parity match`, `committed_exact_tokens/pass = 2`, and `accepted_future_tokens/pass = 1`
