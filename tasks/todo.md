# ANE Decode Throughput Campaign (2026-03-06)

## Plan
- [x] Avenue 1: Multi-layer kernel fusion
  - [x] Re-read prior findings and current fused decode implementation before edits
  - [x] Add failing MIL generation + hardware-gated timing tests for 2-layer packed-cache fusion
  - [x] Record fused single-layer baseline measurement
  - [x] Implement `FusedTwoLayerDecodeGenerator`, `FusedTwoLayerDecodeKernelSet`, and decode-loop support
  - [x] Record post-change measurement and compute per-token delta
  - [x] If 2-layer fusion works, probe 3-layer fusion and document weight-streaming behavior
  - [x] Commit Avenue 1 atomically and append findings to docs
- [ ] Avenue 2: Metal SharedEvent on standard eval path
  - [ ] Re-read VC/shared-event findings before edits
  - [ ] Add failing probe tests for Metal-created shared event attachment and signaling
  - [ ] Record baseline standard-eval behavior
  - [ ] Implement `ane_interop_probe_metal_shared_event()` and Swift wrappers/tests
  - [ ] Record post-probe behavior, measure any pipeline overlap benefit, or abandon on dead-end criteria
  - [ ] Commit or document abandonment atomically
- [ ] Avenue 3: Metal + ANE hybrid decode
  - [ ] Re-read prior decode/runtime findings before edits
  - [ ] Add failing correctness/latency tests for standalone Metal SDPA over IOSurface FP16
  - [ ] Record ANE-only decode baseline for comparison
  - [ ] Implement staged hybrid path: standalone Metal SDPA, decode integration, then attempted overlap
  - [ ] Record post-change latency/effective overlap or abandon on dead-end criteria
  - [ ] Commit or document abandonment atomically
- [ ] Avenue 4: CoreML baseline benchmark
  - [ ] Re-read benchmarking findings before edits
  - [ ] Add/export benchmark harness and model export script tests first where practical
  - [ ] Measure 100+ iteration CoreML decode baseline on `.cpuAndNeuralEngine`
  - [ ] Commit benchmark harness and append denominator numbers to docs
- [ ] Avenue 5: Speculative decoding
  - [ ] Re-read decode + prefill findings before edits
  - [ ] Add failing tests for draft/verify acceptance accounting and benchmark harness
  - [ ] Record direct decode baseline measurement
  - [ ] Implement draft model path, batched verification, and acceptance logic for N=2/4/8
  - [ ] Record effective tokens/sec, acceptance rate, and abandon if accept rate < 50% at N=4
  - [ ] Commit or document abandonment atomically
- [ ] Avenue 6: GCD pipeline with completion handler
  - [ ] Re-read completion-handler findings before edits
  - [ ] Add failing benchmark test for pipelined decode loop
  - [ ] Record sequential fused baseline measurement
  - [ ] Implement `runFusedDecodePipelined()` and benchmark it
  - [ ] Record post-change delta and abandon if savings < 10µs/token
  - [ ] Commit or document abandonment atomically
- [ ] Final validation and reporting
  - [ ] Append results summary table to `docs/fused-decode-and-next-steps.md`
  - [ ] Update project MEMORY with any new ANE facts/gotchas
  - [ ] Run `swift test`
  - [ ] Run hardware-gated tests with `ANE_HARDWARE_TESTS=1 swift test`
  - [ ] Report direct ANE vs CoreML final speedup

## Review
- Avenue 1 review:
  - Built and tested the full packed-cache two-layer candidate first.
  - Hardware compile failed immediately with `_ANECompiler : ANECCompile() FAILED` and `InvalidMILProgram`.
  - Retried with the documented fallback: K/V-only packing plus separate mask inputs.
  - Fallback hit the same `InvalidMILProgram` at compile time.
  - Result: abandoned without a post-change latency measurement; cumulative savings unchanged at `0.000ms/token`.
- Pending. Fill in the remaining avenues with baseline, post-change numbers, dead-end evidence, and commit SHAs.

### Avenue 2 review
- Status: abandoned
- Failure mode: attaching a Metal-backed shared event through `setSharedEvents:` causes the standard eval path to hang on hardware before completion or event value advancement.
- Baseline/post benchmark: not recorded because the path never reached a measurable steady-state eval.
- Savings: +0.000ms/token.
- Cumulative savings after Avenue 2: 0.000ms/token.

### Avenue 3 review
- Status: abandoned for now
- Positive result: zero-copy IOSurface-backed `MTLBuffer` binding works, and standalone Metal SDPA runs correctly.
- Measured standalone Metal attention: mean `0.395252 ms/eval`, median `0.217437 ms/eval` at `heads=12`, `headDim=64`, `seqLen=32` with `mach_absolute_time()` timing.
- Baseline ANE attention on the decode path: `201.516 us/layer` average from the 6-layer decode kernel profile.
- Integration blocker: `DecodeKernelSet.decodeAttnQKV` already performs attention math internally, so a real hybrid path requires a new ANE QKV-only kernel before Metal can be inserted.
- Net effect for this pass: no landed decode savings.

### Avenue 4 review
- Status: measured
- Exporter fix: `scripts/generate_coreml_model.py` now supports `--layers` and `--output`, and produced `benchmarks/models/transformer_6layer.mlpackage`.
- Direct ANE decode median: `2.875 ms/token`.
- CoreML `.cpuAndNeuralEngine` median: `3.007 ms/token`.
- Speedup vs CoreML `.cpuAndNeuralEngine`: `1.05x`.
- Strict fastest-CoreML speedup observed in the same run: `1.04x`.
