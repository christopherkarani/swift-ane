# TODO

- [x] Keep `3e6cced` as the exact two-step architecture checkpoint, `2e49cab` as the compile/init gate evidence, and `13c688b` as the student-sidecar artifact checkpoint.
- [x] Re-run hardware truth with longer-budget standalone or release probes before declaring the compile path permanently blocked; log control compile/init time, two-step compile/init time, committed exact tokens/pass, effective `ms/token`, and exact parity status if the run clears first output.
- [x] Add the smallest direct compile/init instrumentation seam needed to distinguish very long ANE compile latency from a true deadlock or shared-gate stall.
- [x] Commit `espresso-multitoken-probe` as the low-overhead exact hardware truth seam and keep its repeated 1-layer/2-layer/3-layer/6-layer results recoverable in docs and Wax.
- [ ] Commit the fused-pair two-step trunk breakthrough and keep the repeated 2-layer/4-layer wins recoverable in docs and Wax.
- [ ] Attack the next measured bottleneck with one hypothesis: extend fused two-step trunk reuse from pairs to triplets and rerun the 6-layer exact comparison before touching future-head training.
- [ ] If fused two-step trunk work still fails to produce a broader exact win, batch both prepared activations through one verifier-head eval and rerun the same matched-control ladder.
- [ ] Keep attempt logging exhaustive in `docs/fused-decode-and-next-steps.md`, update review notes below after every major result, write Wax session + durable notes after each confirmed result, and flush immediately.

# Review

- Open-ended throughput search resumed from `feat/ane-multitoken` after the student-sidecar checkpoint.
- The previous hardware gate result remains: both compile/init-only seams failed to reach first output within roughly `45s`, so no honest same-session medians were reported from that bounded pass.
- The student-route artifact seam is now in place: `TwoStepStudentCheckpoint`, focused tests passed, and `espresso-train --export-two-step-student` writes a recoverable sidecar artifact without changing the base checkpoint format.
- The new standalone release probe recovered hardware truth outside `xctest`: exact parity held on every measured comparison, `committed_exact_tokens/pass` stayed at `2.0`, and `accepted_future_tokens/pass` stayed at `1.0` on the echo checkpoint family.
- Compile/init is not a hard deadlock in the fresh probe: one 6-layer compile/init-only run measured control `36625.966 ms` versus two-step `812.478 ms`.
- Repeated 1-layer matched-control runs all favored exact two-step: control `1.452750`, `1.768331`, `1.788609 ms/token`; two-step `1.354299`, `1.419352`, `1.484302 ms/token`.
- The deeper scaling boundary is still negative on the current checkpoint family: 2-layer fused-pair was noisy and centered slightly behind control, while 3-layer fused-triplet lost in both repeats and 6-layer fused-triplet lost in the initial run.
- Fusing the exact two-step trunk into pair sessions materially changed the ceiling: repeated 2-layer fused-pair runs now favored exact two-step (`1.534096`, `1.556641 ms/token`) over the matched fused-pair control (`2.124839`, `1.679589 ms/token`) with exact parity and `2.0` committed exact tokens/pass.
- The pair-fused win extends through 4 layers: control `2.195484`, `2.334737 ms/token` versus fused-pair two-step `2.149909`, `2.234477 ms/token`.
- The 6-layer gap narrowed sharply but did not close yet: fused-triplet control `2.146151`, `2.293576 ms/token` versus 6-layer two-step built from fused pairs `2.317794`, `2.529677 ms/token`.
