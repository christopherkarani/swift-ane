# TODO

- [x] Replace the echo-only probe input with a real-checkpoint loading path that uses existing generation weight loaders and keeps the echo path available only as an explicit synthetic mode.
- [x] Add failing tests for benchmark input selection and matched CoreML decode configuration before implementing the new probe surface.
- [x] Extract the minimal shared benchmarking helpers needed for the standalone probe to run exact ANE control, exact two-step ANE, and CoreML decode under one binary and one timing contract.
- [x] Extend `espresso-multitoken-probe` with explicit benchmark input modes, real-checkpoint arguments, and same-session CoreML comparison output.
- [x] Add a one-command reproduction script that builds the release probe, runs fresh-process repeats, captures raw JSON/log files, and prints the exact ANE/CoreML ratio gate.
- [x] Re-run focused tests and the release reproduction path; report exact parity, committed exact tokens/pass, accepted future tokens/pass, effective `ms/token`, and the matched CoreML ratio with repeated medians.
- [x] Update docs, review notes, lessons, and Wax session/durable memory with only the results confirmed in this session; hand off and flush at the checkpoint.

# Review

- The current branch contains a real exact two-step breakthrough, but the strongest `4x` evidence still comes from `espresso-multitoken-probe` on the echo checkpoint family.
- `espresso-multitoken-probe` hardcodes `makeEchoRecurrentGenerationWeights(...)`, so the current public claim does not yet have real-checkpoint external validity.
- The branch already has a real generation-weight loader in `GenerationWeights.load(modelPath:)`, so the next step is integration rather than inventing a second checkpoint path.
- The branch already has CoreML decode benchmarking logic in `GenerationHarnessHardwareTests` and `EspressoBench`; the next step is to reuse that logic inside the standalone probe so ANE and CoreML are measured in one executable and one session.
- A defensible public claim now needs a one-command reproduction harness, raw result artifacts, and repeated fresh-process medians instead of one favorable run against saved baselines.
- `RecurrentGenerationWeightStore` now gives the probe a file-based recurrent inference-weight path, and `MultitokenProbeConfiguration` forces the caller to declare whether the run is synthetic `echo` or recurrent-checkpoint based.
- The new one-command harness (`scripts/reproduce_exact_4x.sh`) built the release probe, regenerated the missing 6-layer CoreML package, and produced five matched same-session ANE/CoreML runs under one executable contract.
- The reproducible matched same-session result on the explicit `echo` input mode is weaker than the earlier saved-baseline claim: median-of-five exact two-step `1.6341614583333333 ms/token` versus matched CoreML `5.86440625 ms/token`, or about `3.6986413138746621x`.
- Exactness stayed intact in the reproducibility run: parity matched on all five runs, `committed_exact_tokens/pass` stayed `2`, and `accepted_future_tokens/pass` stayed `1`.
- The correct public stance from this branch is now narrower: this is a reproducible exact multi-token architectural win on the synthetic echo family, but not yet an honest same-session `4x over CoreML` result.
